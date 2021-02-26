//! Porco is a library for working with and composing probability
//! distributions.
//!
//! The API is inspired by the contents of
//! [Probabilistic Functional Programming in Haskell][paper]
//! but with naming conventions that match those of [`Option`] and [`Result`]
//! (such as [`Option::and_then`]).
//!
//! ```rust
//! # use porco::{Probability, Distribution};
//! # #[derive(Debug, PartialEq)]
//! enum Coin {
//!     Heads,
//!     Tails,
//! }
//!
//! impl Coin {
//!     fn flip() -> Distribution<Coin> {
//!         Distribution::uniform([Coin::Heads, Coin::Tails])
//!     }
//! }
//!
//! let coin = Coin::flip();
//! assert_eq!(coin.pmf(&Coin::Heads), Probability(0.5));
//! ```
//!
//! You can compose various operations over `Distribution`s using combinators
//! like [`Distribution::map`], [`Distribution::and_then`], and
//! [`Distribution::given`].
//!
//! ```rust
//! # use porco::{Probability, Distribution};
//! # #[derive(Debug, PartialEq)]
//! # enum Coin {
//! #     Heads,
//! #     Tails,
//! # }
//! # impl Coin {
//! #     const OUTCOMES: [Coin; 2] = [Coin::Heads, Coin::Tails];
//! #     fn flip() -> Distribution<Coin> {
//! #         Distribution::uniform(Coin::OUTCOMES)
//! #     }
//! # }
//! fn reflip_if_tails(coin: Coin) -> Distribution<Coin> {
//!     match coin {
//!         Coin::Heads => Distribution::always(Coin::Heads),
//!         Coin::Tails => Coin::flip(),
//!     }
//! }
//!
//! let coin = Coin::flip().and_then(reflip_if_tails);
//! assert_eq!(coin.pmf(&Coin::Heads), Probability(0.75));
//! ```
//!
//! You can also compute summary statistics of random variables.
//!
//! ```rust
//! # use porco::{Probability, Distribution};
//! let die = Distribution::uniform([1, 2, 3, 4, 5, 6]);
//! let ev = die.given(|&v| v <= 4).expectation();
//! assert_eq!(ev, 2.5);
//! ```
//!
//! [paper]: https://web.engr.oregonstate.edu/~erwig/papers/PFP_JFP06.pdf
mod assoc_list;
mod dist;
mod prob;

pub use dist::Distribution;
pub use prob::Probability;
