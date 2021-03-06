# porco

[<img alt="docs.rs" src="https://docs.rs/porco/badge.svg">](https://docs.rs/porco)

Composable probability distributions.

## Examples

Create simple probability distributions.

```rust
enum Coin {
    Heads,
    Tails,
}
 
impl Coin {
    fn flip() -> Distribution<Coin> {
        Distribution::uniform([Coin::Heads, Coin::Tails])
    }
}
 
let coin = Coin::flip();
assert_eq!(coin.pmf(&Coin::Heads), Probability(0.5));
```

Compose operations over distributions using combinators.

```rust
fn reflip_if_tails(coin: Coin) -> Distribution<Coin> {
    match coin {
        Coin::Heads => Distribution::always(Coin::Heads),
        Coin::Tails => Coin::flip(),
    }
}
 
let coin = Coin::flip().and_then(reflip_if_tails);
assert_eq!(coin.pmf(&Coin::Heads), Probability(0.75));
```

Compute summary statistics of random variables.

```rust
let die = Distribution::uniform([1, 2, 3, 4, 5, 6]);
let ev = die.given(|&v| v <= 4).expectation();
assert_eq!(ev, 2.5);
```
