use crate::{assoc_list::AssociationExt, prob::Probability};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Distribution<T>(Vec<(T, Probability)>);

impl<T> Distribution<T>
where
    T: Eq,
{
    pub fn new<V: Into<Vec<(T, Probability)>>>(v: V) -> Distribution<T> {
        Distribution(v.into()).regroup()
    }

    pub fn always(t: T) -> Distribution<T> {
        Distribution(vec![(t, Probability::ONE)])
    }

    pub fn uniform<V: Into<Vec<T>>>(outcomes: V) -> Distribution<T> {
        let outcomes = outcomes.into();
        let p = Probability(1.0 / outcomes.len() as f64);
        Distribution::from(outcomes.into_iter().map(|t| (t, p)).collect::<Vec<_>>())
    }

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

    pub fn pmf(&self, t: &T) -> Probability {
        *self.0.get_(t).unwrap_or(&Probability::ZERO)
    }
}

impl<T> Distribution<Distribution<T>>
where
    T: Eq,
{
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
        fn flip_again_if_tails(coin: Coin) -> Distribution<Coin> {
            match coin {
                Coin::Heads => Distribution::always(Coin::Heads),
                Coin::Tails => Coin::flip(),
            }
        }
        let d1 = Coin::flip().map(flip_again_if_tails).flatten();
        let d2 = Coin::flip().and_then(flip_again_if_tails);
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
