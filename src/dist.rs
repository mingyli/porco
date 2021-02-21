use crate::{assoc_list::AssociationExt, Probability};

/// [`Distribution<T>`] is a discrete probability distribution over
/// the set of outcomes `T`.
///
/// See the [module level documentation for an overview](crate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Distribution<T>(Vec<(T, Probability)>);

impl<T> Distribution<T>
where
    T: Eq,
{
    /// Create a distribution using given outcome probabilities.
    pub fn new<V: Into<Vec<(T, Probability)>>>(v: V) -> Distribution<T> {
        Distribution(v.into()).regroup()
    }

    /// Create a distribution where the given outcome always occurs.
    pub fn always(t: T) -> Distribution<T> {
        Distribution(vec![(t, Probability::ONE)])
    }

    /// Create a uniform distribution over a collection of outcomes.
    pub fn uniform<V: Into<Vec<T>>>(outcomes: V) -> Distribution<T> {
        let outcomes = outcomes.into();
        let p = Probability(1.0 / outcomes.len() as f64);
        Distribution::from(outcomes.into_iter().map(|t| (t, p)).collect::<Vec<_>>())
    }

    /// Convert a `Distribution<T>` into a `Distribution<U>` by mapping
    /// outcomes in `T` to outcomes in `U`.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Debug, PartialEq, Eq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// let dist = Distribution::uniform([0, 1, 2, 3]).map(|v| {
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
        U: Eq,
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
    /// # #[derive(Debug, PartialEq, Eq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// fn roll_a_die_if_heads(coin: Coin) -> Distribution<Option<u8>> {
    ///     match coin {
    ///         Coin::Heads => Distribution::uniform([Some(1), Some(2), Some(3), Some(4)]),
    ///         Coin::Tails => Distribution::always(None),
    ///     }
    /// }
    ///
    /// let dist = Distribution::uniform([Coin::Heads, Coin::Tails])
    ///     .and_then(roll_a_die_if_heads);
    /// assert_eq!(dist.pmf(&None), Probability(0.5));
    /// assert_eq!(dist.pmf(&Some(2)), Probability(0.125));
    /// ```
    ///
    /// [`Distribution::and_then`] can also be used to construct joint distributions.
    ///
    /// ```
    /// # use porco::{Distribution, Probability};
    /// # #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// # impl Coin {
    /// #     fn flip() -> Distribution<Coin> {
    /// #         Distribution::uniform([Coin::Heads, Coin::Tails])
    /// #     }
    /// # }
    /// fn flip_another(coin: Coin) -> Distribution<(Coin, Coin)> {
    ///     Distribution::uniform([(coin, Coin::Heads), (coin, Coin::Tails)])
    /// }
    ///
    /// let two_coins = Coin::flip().and_then(flip_another);
    /// assert_eq!(two_coins.pmf(&(Coin::Heads, Coin::Heads)), Probability(0.25));
    /// ```
    pub fn and_then<F, U>(self, f: F) -> Distribution<U>
    where
        U: Eq,
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
            let p = p + match ass.get_(&t) {
                Some(prob) => *prob,
                None => Probability::ZERO,
            };
            ass.insert_(t, p);
        }
        Distribution(ass)
        /*
         * TODO: Rewrite after Entry API is implemented.
         * for (t, p) in self.0 {
         *     ass.entry(t).and_modify(|e| *e = *e + p).or_insert(p);
         * }
         */
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
    /// # #[derive(Debug, PartialEq, Eq)]
    /// # enum Coin {
    /// #     Heads,
    /// #     Tails,
    /// # }
    /// let die = Distribution::uniform([1, 2, 3, 4, 5, 6]);
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
        *self.0.get_(t).unwrap_or(&Probability::ZERO)
    }
}

impl<T> Distribution<Distribution<T>>
where
    T: Eq,
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

impl<T> From<Vec<(T, Probability)>> for Distribution<T>
where
    T: Eq,
{
    fn from(v: Vec<(T, Probability)>) -> Self {
        Distribution::new(v)
    }
}

impl<T, const N: usize> From<[(T, Probability); N]> for Distribution<T>
where
    T: Eq,
{
    fn from(s: [(T, Probability); N]) -> Self {
        Distribution::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    enum Coin {
        Heads,
        Tails,
    }
    impl Coin {
        const OUTCOMES: [Coin; 2] = [Coin::Heads, Coin::Tails];

        fn flip() -> Distribution<Coin> {
            Distribution::uniform(Coin::OUTCOMES)
        }
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    enum Die {
        One = 1,
        Two = 2,
        Three = 3,
        Four = 4,
        Five = 5,
        Six = 6,
    }
    impl Die {
        const OUTCOMES: [Die; 6] = [
            Die::One,
            Die::Two,
            Die::Three,
            Die::Four,
            Die::Five,
            Die::Six,
        ];

        fn roll() -> Distribution<Die> {
            Distribution::uniform(Die::OUTCOMES)
        }
    }

    #[test]
    fn test_coin() {
        let coin_flip = Distribution::uniform(Coin::OUTCOMES);
        assert_eq!(
            coin_flip,
            Distribution::from([
                (Coin::Heads, Probability(0.5)),
                (Coin::Tails, Probability(0.5))
            ])
        );
    }

    #[test]
    fn test_and_then() {
        fn reflip_if_tails(coin: Coin) -> Distribution<Coin> {
            match coin {
                Coin::Heads => Distribution::always(Coin::Heads),
                Coin::Tails => Coin::flip(),
            }
        }
        let d1 = Coin::flip().map(reflip_if_tails).flatten();
        let d2 = Coin::flip().and_then(reflip_if_tails);
        assert_eq!(d1, d2);
        assert_eq!(d1.pmf(&Coin::Tails), (Probability(0.25)));
        assert_eq!(d1.pmf(&Coin::Heads), (Probability(0.75)));
    }

    #[test]
    fn test_die() {
        let five_or_six = Die::roll().map(|die| matches!(die, Die::Five | Die::Six));
        assert_eq!(five_or_six.pmf(&true), (Probability(1.0 / 3.0)));
    }

    #[test]
    fn test_two_dice() {
        fn roll_another_die(die: Die) -> Distribution<(Die, Die)> {
            Die::roll().and_then(|d| Distribution::always((die, d)))
        }
        let two_dice = Die::roll().and_then(roll_another_die);
        let sum_two_dice_eight_or_nine = two_dice
            .map(|(d1, d2)| d1 as u8 + d2 as u8)
            .map(|s| s == 8 || s == 9);
        assert_eq!(sum_two_dice_eight_or_nine.pmf(&true), Probability(0.25));
    }

    #[test]
    fn test_given() {
        fn less_than_three(die: &Die) -> bool {
            matches!(die, Die::One | Die::Two)
        }
        let die = Die::roll();
        let die_with_more_knowledge = die.given(less_than_three);
        assert_eq!(die_with_more_knowledge.pmf(&Die::One), Probability(0.5));
    }
}
