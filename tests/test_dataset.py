import torch

from training.dataset import NextTokenDataset, create_dataloaders, split_token_stream


def test_dataset_creates_shifted_input_target_pairs():
    dataset = NextTokenDataset(torch.arange(10), block_size=4)
    x, y = dataset[0]

    assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4]))
    assert len(dataset) == 6


def test_dataset_samples_have_equal_sequence_length():
    x, y = NextTokenDataset(torch.arange(20), block_size=6)[3]

    assert x.shape == y.shape == (6,)


def test_token_stream_split_is_deterministic_and_ordered():
    tokens = torch.arange(30)
    first_split = split_token_stream(tokens)
    second_split = split_token_stream(tokens)

    assert all(torch.equal(a, b) for a, b in zip(first_split, second_split))
    assert torch.equal(first_split[0], tokens[:27])
    assert torch.equal(first_split[1], tokens[27:])


def test_dataloader_returns_batch_and_sequence_shapes():
    train_loader, validation_loader = create_dataloaders(
        torch.arange(40), torch.arange(40, 60), block_size=4, batch_size=3
    )
    train_x, train_y = next(iter(train_loader))
    validation_x, validation_y = next(iter(validation_loader))

    assert train_x.shape == train_y.shape == (3, 4)
    assert validation_x.shape == validation_y.shape == (3, 4)
