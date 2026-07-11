import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import get_config
from dataset import BiosignalDataset
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn1d", "resnet"], default="cnn1d")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target-class", type=int, default=6)
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--max-check", type=int, default=4000)
    return parser.parse_args()


def load_model(config, checkpoint, device):
    model = get_model(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_target_layer(model, model_name):
    if model_name == "cnn1d":
        return model.conv5
    return model.block4.conv2


class GradCam1D:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.activations = None
        self.gradients = None

        self.forward_hook = layer.register_forward_hook(self.save_activation)
        self.backward_hook = layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        score = output[0, class_idx]
        score.backward()

        # average gradients over time
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def attention_map(model, x):
    with torch.no_grad():
        _ = model(x)

    weights = model.attention.last_weights

    weights = weights.transpose(1, 2)
    weights = F.interpolate(weights, size=x.shape[-1], mode="linear", align_corners=False)
    weights = weights.squeeze().detach().cpu().numpy()

    if weights.max() > weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights = np.zeros_like(weights)

    return weights


def find_examples(model, dataset, device, class_idx, max_check):
    correct = None
    wrong = None

    for i in range(min(len(dataset), max_check)):
        item = dataset[i]
        x = item["signal"].unsqueeze(0).to(device)
        y = item["label"].numpy()

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0]

        pred = (prob >= 0.5).astype(np.float32)
        true_value = y[class_idx]
        pred_value = pred[class_idx]

        if correct is None and true_value == 1 and pred_value == 1:
            correct = (i, item, prob)

        if wrong is None and true_value != pred_value:
            wrong = (i, item, prob)

        if correct is not None and wrong is not None:
            break

    return correct, wrong


def plot_xai(signal, heatmap, title, out_path, lead_idx):
    lead_signal = signal[lead_idx]
    t = np.arange(len(lead_signal))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, lead_signal, color="black", linewidth=1)
    ax.imshow(
        heatmap.reshape(1, -1),
        aspect="auto",
        cmap="jet",
        alpha=0.35,
        extent=[0, len(lead_signal), lead_signal.min(), lead_signal.max()],
    )
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel(f"Lead {lead_idx + 1}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def explain_sample(model, model_name, item, prob, class_idx, lead_idx, out_dir, tag):
    signal = item["signal"].numpy()
    x = item["signal"].unsqueeze(0).to(next(model.parameters()).device)

    # Grad-CAM on the last CNN layer
    gradcam = GradCam1D(model, get_target_layer(model, model_name))
    cam = gradcam(x, class_idx)
    gradcam.remove()

    class_name = f"class{class_idx + 1}"
    plot_xai(
        signal,
        cam,
        f"{model_name} Grad-CAM ({tag}, {class_name}, prob={prob[class_idx]:.3f})",
        os.path.join(out_dir, f"{model_name}_{class_name}_{tag}_gradcam.png"),
        lead_idx,
    )

    if model_name == "cnn1d":
        attn = attention_map(model, x)
        plot_xai(
            signal,
            attn,
            f"{model_name} Attention ({tag}, {class_name}, prob={prob[class_idx]:.3f})",
            os.path.join(out_dir, f"{model_name}_{class_name}_{tag}_attention.png"),
            lead_idx,
        )


def main():
    args = parse_args()
    config = get_config()
    config["model"]["name"] = args.model

    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(config["paths"]["checkpoints"], f"{args.model}_fold0.pt")

    out_dir = os.path.join(config["paths"]["outputs"], "xai")
    os.makedirs(out_dir, exist_ok=True)

    class_idx = args.target_class - 1
    lead_idx = args.lead - 1

    dataset = BiosignalDataset(config)
    model = load_model(config, checkpoint, device)
    correct, wrong = find_examples(model, dataset, device, class_idx, args.max_check)

    if correct is not None:
        idx, item, prob = correct
        print(f"Correct example: sample {idx}")
        explain_sample(model, args.model, item, prob, class_idx, lead_idx, out_dir, "correct")
    else:
        print("No correct positive example found.")

    if wrong is not None:
        idx, item, prob = wrong
        print(f"Wrong example: sample {idx}")
        explain_sample(model, args.model, item, prob, class_idx, lead_idx, out_dir, "wrong")
    else:
        print("No wrong example found.")

    print(f"Saved XAI plots to {out_dir}")


if __name__ == "__main__":
    main()
