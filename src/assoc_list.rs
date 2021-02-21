pub trait AssociationExt<K, V> {
    fn insert_(&mut self, k: K, v: V) -> Option<V>;
    fn get_(&self, k: &K) -> Option<&V>;
    fn find_(&self, k: &K) -> Option<usize>;
}

impl<K, V> AssociationExt<K, V> for Vec<(K, V)>
where
    K: Eq,
{
    fn insert_(&mut self, k: K, v: V) -> Option<V> {
        let result = match self.find_(&k) {
            Some(index) => Some(self.swap_remove(index).1),
            None => None,
        };
        self.push((k, v));
        result
    }

    fn get_(&self, key: &K) -> Option<&V> {
        match self.find_(key) {
            Some(index) => self.get(index).map(|(_, v)| v),
            None => None,
        }
    }

    fn find_(&self, key: &K) -> Option<usize> {
        for (i, (k, _)) in self.iter().enumerate() {
            if key == k {
                return Some(i);
            }
        }
        None
    }
}

