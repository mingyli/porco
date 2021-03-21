use std::iter::FromIterator;

use assoc::AssocExt;

use crate::Probability;

/// [`Distribution<T>`] is a discrete probability distribution over
/// the set of outcomes `T`.
///
/// See the [module level documentation for an overview](crate).
///
/// The underlying implementation of a `Distribution<T>` is an associative
/// array `Vec<T, Probability>` through the [`assoc`] crate.
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution<T>(Vec<(T, Probability)>);

impl<T> Distribution<T>
where
    T: PartialEq,
{
    /// Create a distribution using given outcome probabilities.
    pub fn new<I: IntoIterator<Item = (T, Probability)>>(iter: I) -> Distribution<T> {
        Distribution(iter.into_iter().collect()).regroup()
    }

    /// Create a distribution where the given outcome always occurs.
    pub fn always(t: T) -> Distribution<T> {
        Distribution(vec![(t, Probability::ONE)])
    }

    /// Create a uniform distribution over a collection of outcomes.
    pub fn uniform<I: IntoIterator<Item = T>>(iter: I) -> Distribution<T> {
        let outcomes: Vec<_> = iter.into_iter().collect();
        let p = Probability(1.0 / outcomes.len() as f64);
        Distribution::from(outcomes.into_iter().map(|t| (t, p)).collect::<Vec<_>>())
    }

    /// Convert a `Distribution<T>` into a `Distribution<U>` by mapping
    /// outcomes in `T` to outcomes in `U`.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Debug, PartialEq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// let dist = Distribution::uniform(vec![0, 1, 2, 3]).map(|v| {
    ///     if v == 3 {
    ///         Coin::Heads
    ///     } else {
    ///         Coin::Tails
    ///     }
    /// });
    /// assert_eq!(dist.pmf(&Coin::Heads), Probability(0.25));
    /// assert_eq!(dist.pmf(&Coin::Tails), Probability(0.75));
    /// ```
    pub fn map<F, U>(self, f: F) -> Distribution<U>
    where
        U: PartialEq,
        F: Fn(T) -> U,
    {
        Distribution::from(
            self.0
                .into_iter()
                .map(|(t, p)| (f(t), p))
                .collect::<Vec<_>>(),
        )
    }

    /// Convert a `Distribution<T>` into a `Distribution<U>` by mapping
    /// outcomes in `T` to distributions over `U`.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Debug, PartialEq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// # impl Coin {
    /// #     fn flip() -> Distribution<Coin> {
    /// #         Distribution::uniform(vec![Coin::Heads, Coin::Tails])
    /// #     }
    /// # }
    /// fn roll_a_die_if_heads(coin: Coin) -> Distribution<Option<u8>> {
    ///     match coin {
    ///         Coin::Heads => Distribution::uniform(vec![Some(1), Some(2), Some(3), Some(4)]),
    ///         Coin::Tails => Distribution::always(None),
    ///     }
    /// }
    ///
    /// let dist = Coin::flip().and_then(roll_a_die_if_heads);
    /// assert_eq!(dist.pmf(&None), Probability(0.5));
    /// assert_eq!(dist.pmf(&Some(2)), Probability(0.125));
    /// ```
    ///
    /// [`Distribution::and_then`] can also be used to construct joint distributions.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Copy, Clone, Debug, PartialEq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// # impl Coin {
    /// #     fn flip() -> Distribution<Coin> {
    /// #         Distribution::uniform(vec![Coin::Heads, Coin::Tails])
    /// #     }
    /// # }
    /// fn flip_another(coin: Coin) -> Distribution<(Coin, Coin)> {
    ///     Distribution::uniform(vec![(coin, Coin::Heads), (coin, Coin::Tails)])
    /// }
    ///
    /// let two_coins = Coin::flip().and_then(flip_another);
    /// assert_eq!(two_coins.pmf(&(Coin::Heads, Coin::Heads)), Probability(0.25));
    /// ```
    pub fn and_then<F, U>(self, f: F) -> Distribution<U>
    where
        U: PartialEq,
        F: Fn(T) -> Distribution<U>,
    {
        Distribution::from(
            self.0
                .into_iter()
                .map(|(t, p)| (f(t), p))
                .flat_map(|(dist, p)| dist.0.into_iter().map(move |(t, p2)| (t, p * p2)))
                .collect::<Vec<_>>(),
        )
    }

    fn regroup(self) -> Distribution<T> {
        let mut ass = Vec::new();
        for (t, p) in self.0 {
            ass.entry(t).and_modify(|e| *e = *e + p).or_insert(p);
        }
        Distribution(ass)
    }

    fn normalize(self) -> Distribution<T> {
        let factor: f64 = self.0.iter().map(|(_, p)| p.0).sum();
        Distribution::from(
            self.0
                .into_iter()
                .map(|(t, p)| (t, p / factor))
                .collect::<Vec<_>>(),
        )
    }

    /// Create a distribution from a distribution conditioned on an event occurring.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Debug, PartialEq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// let die = Distribution::uniform(vec![1, 2, 3, 4, 5, 6]);
    /// let die_given_less_than_three = die.given(|&v| v < 3);
    /// assert_eq!(die_given_less_than_three.pmf(&1), Probability(0.5));
    /// ```
    pub fn given<F>(self, condition: F) -> Distribution<T>
    where
        F: Fn(&T) -> bool,
    {
        Distribution::from(
            self.0
                .into_iter()
                .filter(|(t, _)| condition(t))
                .collect::<Vec<_>>(),
        )
        .normalize()
    }

    /// Get the probability of an outcome occurring from the probability mass function.
    pub fn pmf(&self, t: &T) -> Probability {
        *self.0.get(t).unwrap_or(&Probability::ZERO)
    }
}

impl<T> Distribution<T>
where
    T: Into<f64> + Clone,
{
    /// Compute the expectation of a random variable.
    ///
    /// A random variable is a mapping from outcomes to real values.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Debug, PartialEq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// let biased_coin = Distribution::from([
    ///     (Coin::Heads, Probability(0.25)),
    ///     (Coin::Tails, Probability(0.75)),
    /// ]);
    /// let ev = biased_coin
    ///     .map(|coin| match coin {
    ///         Coin::Heads => 1,
    ///         Coin::Tails => 0,
    ///     })
    ///     .expectation();
    /// assert_eq!(ev, 0.25);
    /// ```
    pub fn expectation(&self) -> f64 {
        self.0.iter().map(|(t, p)| t.clone().into() * p.0).sum()
    }
}

impl<T> Distribution<Distribution<T>>
where
    T: PartialEq,
{
    /// Convert a `Distribution<Distribution<T>>` into a `Distribution<T>`.
    ///
    /// A `Distribution<Distribution<T>>` can be interpreted as a sequence of
    /// two experiments, where the outcome of the first informs what experiment
    /// is conducted second.
    pub fn flatten(self) -> Distribution<T> {
        self.and_then(std::convert::identity)
    }
}

impl<T> FromIterator<(T, f64)> for Distribution<T>
where
    T: PartialEq,
{
    /// ```rust
    /// use porco::Distribution;
    ///
    /// let dist: Distribution<&str> = vec![("a", 0.4), ("b", 0.6)].into_iter().collect();
    /// ```
    fn from_iter<I: IntoIterator<Item = (T, f64)>>(iter: I) -> Self {
        let v: Vec<_> = iter.into_iter().map(|(t, p)| (t, Probability(p))).collect();
        Distribution::new(v)
    }
}

impl<T> From<Vec<(T, Probability)>> for Distribution<T>
where
    T: PartialEq,
{
    fn from(v: Vec<(T, Probability)>) -> Self {
        Distribution::new(v)
    }
}

impl<T, const N: usize> From<[(T, Probability); N]> for Distribution<T>
where
    T: PartialEq,
{
    fn from(s: [(T, Probability); N]) -> Self {
        use std::array;

        Distribution::new(array::IntoIter::new(s))
    }
}
