import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.triangle import Triangle
from src.models.individual import Individual
from src.strategies.mutation.MultiGenLimitedMutation import MultiGenLimitedMutation


def _make_triangle(idx: int) -> Triangle:
    return Triangle((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (idx, idx, idx, idx))


def test_multigen_mutation_selects_subset_and_preserves_parents():
    """Test that MultiGenMutation only mutates a subset of triangles and preserves parent."""
    t0, t1, t2, t3, t4 = (_make_triangle(i) for i in range(5))
    original = Individual([t0, t1, t2, t3, t4])
    original_triangles = list(original.triangles)
    
    # Use fixed seed for reproducible test
    mutator = MultiGenLimitedMutation(min_genes=2, max_genes=3, rng=random.Random(42))
    mutated = mutator.mutate(original)
    
    # Should have same number of triangles
    assert len(mutated.triangles) == 5
    
    # Original should be unchanged
    assert original.triangles == original_triangles
    
    # Some triangles should be unchanged (not all mutated)
    unchanged_count = 0
    changed_count = 0
    for orig, mut in zip(original.triangles, mutated.triangles):
        if (orig.p1 == mut.p1 and orig.p2 == mut.p2 and 
            orig.p3 == mut.p3 and orig.color == mut.color):
            unchanged_count += 1
        else:
            changed_count += 1
    
    # Should have mutated between 2-3 triangles (max_genes=3)
    assert 2 <= changed_count <= 3
    assert unchanged_count >= 2  # At least 2 should remain unchanged
    assert unchanged_count + changed_count == 5


def test_multigen_mutation_respects_min_max_genes():
    """Test that MultiGenMutation respects min_genes and max_genes bounds."""
    triangles = [_make_triangle(i) for i in range(10)]
    original = Individual(triangles)
    
    mutator = MultiGenLimitedMutation(min_genes=1, max_genes=1, rng=random.Random(123))
    mutated = mutator.mutate(original)
    
    # Count changes
    changed_count = 0
    for orig, mut in zip(original.triangles, mutated.triangles):
        if not (orig.p1 == mut.p1 and orig.p2 == mut.p2 and 
                orig.p3 == mut.p3 and orig.color == mut.color):
            changed_count += 1
    
    # Should have mutated exactly 1 triangle (min_genes=max_genes=1)
    assert changed_count == 1


def test_multigen_mutation_handles_edge_cases():
    """Test MultiGenMutation with edge cases."""
    # Empty individual
    empty = Individual([])
    mutated_empty = MultiGenLimitedMutation().mutate(empty)
    assert len(mutated_empty.triangles) == 0
    
    # Single triangle
    single = Individual([_make_triangle(0)])
    mutated_single = MultiGenLimitedMutation(min_genes=1, max_genes=1).mutate(single)
    assert len(mutated_single.triangles) == 1
    # Should be mutated (different from original)
    assert mutated_single.triangles[0] != single.triangles[0]