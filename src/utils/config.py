from dataclasses import dataclass, field, asdict

@dataclass
class Config:
# add below with type hint with default values
    learning_rate:float = 5e-4
    n_epochs:int = 5
    n_steps:int = 1000
    batch_size:int = 64
    block_size:int = 64
    device:str = 'mps'
    n_embed:int = 384
    n_heads:int = 8
    # Note this allows head_size to be calculated based on n_embed and n_heads in __post_init__ and show up in __repr__ 
    head_size:int = field(init=False)
    n_blocks:int = 3
    # Accoridng to paper, output of each sub-layer, before it is added to the
    # sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings
    dropout:float = 0.1

    def __post_init__(self):
        self.head_size:int = self.n_embed//self.n_heads

    def to_dict(self):
        return asdict(self)

