import argparse
import torch
from mimo import MimoUNet, apply_input_transform
from loss import MCCE, MCSM


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, num_channels, height, width):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return dict(
            input=torch.randn(self.num_channels, self.height, self.width),
            target=torch.randint(0, 2, (self.height, self.width)),
            weight=torch.rand(self.height, self.width)
        )


def main(
    input_channels: int,
    output_channels: int,
    num_subnetworks: int,
    filter_base_count: int,
    input_repetition_probability: float,
    batch_size: int,
    device: str,
):
    model = MimoUNet(
        in_channels=input_channels,
        out_channels=output_channels,
        num_subnetworks=num_subnetworks,
        filter_base_count=filter_base_count
    ).to(device)
    model.train()

    train_ds = ExampleDataset(20, input_channels, 100, 50)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = ExampleDataset(10, input_channels, 100, 50)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_ds = ExampleDataset(10, input_channels, 100, 50)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    criterion = MCCE(num_samples=50)
    mc_softmax = MCSM(num_samples=50, log=True)

    for epoch in range(10):
        model.train()
        for batch in train_dl:
            inputs, targets, weights = apply_input_transform(
                image=batch['input'],
                label=batch['target'],
                mask=batch['weight'],
                num_subnetworks=num_subnetworks,
                input_repetition_probability=input_repetition_probability,
                batch_repetitions=1,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            prediction = model(inputs)
            logits = prediction[:, :, :2]
            std = torch.exp(prediction[:, :, 2:])

            M = prediction.shape[1]
            loss = 0

            for i in range(M):
                loss += criterion(
                    logits=logits[:, i], 
                    std=std[:, i], 
                    labels=targets[:, i].long(),
                    weight=weights[:, i],
                ) / M

            

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_dl:
                inputs, targets, weights = apply_input_transform(
                    image=batch['input'],
                    label=batch['target'],
                    mask=batch['weight'],
                    num_subnetworks=num_subnetworks,
                    input_repetition_probability=input_repetition_probability,
                    batch_repetitions=1,
                )
                inputs = inputs.to(device)
                targets = targets.to(device)
                weights = weights.to(device)

                prediction = model(inputs)
                logits = prediction[:, :, :2]
                std = torch.exp(prediction[:, :, 2:])

                M = prediction.shape[1]
                for i in range(M):
                    val_loss += criterion(
                        logits=logits[:, i], 
                        std=std[:, i], 
                        labels=targets[:, i].long(),
                        weight=weights[:, i],
                    ) / M

            print(f'Epoch {epoch} - Validation Loss: {val_loss / len(val_dl)}')

        
        model.eval()

    # Evaluate the model
    with torch.no_grad():
        for batch in test_dl:
            inputs, targets, weights = apply_input_transform(
                image=batch['input'],
                label=batch['target'],
                mask=batch['weight'],
                num_subnetworks=num_subnetworks,
                input_repetition_probability=input_repetition_probability,
                batch_repetitions=1,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            prediction = model(inputs)
            logits = prediction[:, :, :2]
            std = torch.exp(prediction[:, :, 2:])

            mc_softmax = MCSM(num_samples=100, log=False)

            # [*batch, M, C, H, W]
            probabilities = mc_softmax(logits, std)
       
            # [*batch, C, H, W]
            probabilities = probabilities.mean(axis=1)

            print(probabilities.shape)
            # todo: use this to compute uncertainty metrics, etc...
            # `probabilities` should contain well calibrated probabilities for each pixel 
            # and contain both aleatoric and epistemic uncertainty



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=4)
    parser.add_argument('--num_subnetworks', type=int, default=3)
    parser.add_argument('--filter_base_count', type=int, default=30)
    parser.add_argument('--input_repetition_probability', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(
        input_channels=args.input_channels,
        output_channels=args.output_channels,
        num_subnetworks=args.num_subnetworks,
        filter_base_count=args.filter_base_count,
        input_repetition_probability=args.input_repetition_probability,
        batch_size=args.batch_size,
        device=args.device,
    )
