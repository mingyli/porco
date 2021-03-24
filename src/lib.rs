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
//!         Distribution::uniform(vec![Coin::Heads, Coin::Tails])
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
//! #     fn flip() -> Distribution<Coin> {
//! #         Distribution::uniform(vec![Coin::Heads, Coin::Tails])
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
//! You can also manipulate random variables and compute summary statistics.
//!
//! ```rust
//! # use porco::{Probability, Distribution};
//! let die = Distribution::uniform(vec![1, 2, 3, 4, 5, 6]);
//! let ev = die.given(|&v| v <= 4).expectation();
//! assert_eq!(ev, 2.5);
//!
//! fn two_sided_die() -> Distribution<u8> {
//!     Distribution::uniform(vec![1, 2])
//! }
//!
//! let x = two_sided_die();
//! let y = two_sided_die();
//! let sum = x.convolve(y);
//! assert_eq!(sum.pmf(&2), Probability(0.25));
//! assert_eq!(sum.pmf(&3), Probability(0.5));
//! ```
//!
//! [paper]: https://web.engr.oregonstate.edu/~erwig/papers/PFP_JFP06.pdf
#![feature(array_value_iter)]
mod dist;
mod prob;

pub use dist::Distribution;
pub use prob::Probability;
