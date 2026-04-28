class DisjointSet:
    """Disjoint Set Union (Union-Find) data structure.

    Maintains a collection of disjoint sets over the integers 0..n-1,
    supporting near-constant-time union and find operations via
    path compression and union by rank.

    The amortized time complexity per operation is O(α(n)), where α
    is the inverse Ackermann function (effectively constant for any
    practical n).

    Example:
        >>> dsu = DSU(5)
        >>> dsu.union(0, 1)
        True
        >>> dsu.union(1, 2)
        True
        >>> dsu.find(0) == dsu.find(2)
        True
        >>> dsu.find(0) == dsu.find(3)
        False
    """

    def __init__(self, n):
        """Initialize n singleton sets: {0}, {1}, ..., {n-1}.

        Args:
            n: The number of elements. Elements are labeled 0..n-1.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Return the representative (root) of the set containing x.

        Two elements belong to the same set if and only if they have
        the same representative. Applies path compression: every node
        traversed is re-parented directly to its grandparent, flattening
        the tree to speed up future queries.

        Args:
            x: The element to look up.

        Returns:
            The root element of x's set.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, *elements):
        """Merge the sets containing all given elements into one.

        Accepts any number of elements and unions them all together.
        Uses union by rank: shallower trees are attached under the root
        of the deeper one, keeping the combined tree shallow.

        Args:
            *elements: Two or more elements to merge into a single set.
                Calling with 0 or 1 elements is a no-op and returns False.

        Returns:
            True if any merging occurred (i.e., at least two of the
            elements were in different sets); False if all elements
            were already in the same set or fewer than 2 were given.
        """
        if len(elements) < 2:
            return False

        merged = False
        rx = self.find(elements[0])
        for y in elements[1:]:
            ry = self.find(y)
            if rx == ry:
                continue
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            merged = True

        return merged
