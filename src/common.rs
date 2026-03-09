use std::hash::{Hash, Hasher};
use std::ops::{Index, Range};

use crate::algorithms::utils::IdentifyDistinct;
use crate::algorithms::{diff_deadline, Capture, Compact, Replace};
use crate::deadline_support::Instant;
use crate::{Algorithm, DiffOp};

/// Creates a diff between old and new with the given algorithm capturing the ops.
///
/// This is like [`diff`](crate::algorithms::diff) but instead of using an
/// arbitrary hook this will always use [`Compact`] + [`Replace`] + [`Capture`]
/// and return the captured [`DiffOp`]s.
pub fn capture_diff<Old, New>(
    alg: Algorithm,
    old: &Old,
    old_range: Range<usize>,
    new: &New,
    new_range: Range<usize>,
) -> Vec<DiffOp>
where
    Old: Index<usize> + ?Sized,
    New: Index<usize> + ?Sized,
    Old::Output: Hash + Eq + Ord,
    New::Output: PartialEq<Old::Output> + Hash + Eq + Ord,
{
    capture_diff_deadline(alg, old, old_range, new, new_range, None)
}

/// Creates a diff between old and new with the given algorithm capturing the ops.
///
/// Works like [`capture_diff`] but with an optional deadline.
pub fn capture_diff_deadline<Old, New>(
    alg: Algorithm,
    old: &Old,
    old_range: Range<usize>,
    new: &New,
    new_range: Range<usize>,
    deadline: Option<Instant>,
) -> Vec<DiffOp>
where
    Old: Index<usize> + ?Sized,
    New: Index<usize> + ?Sized,
    Old::Output: Hash + Eq + Ord,
    New::Output: PartialEq<Old::Output> + Hash + Eq + Ord,
{
    let mut d = Compact::new(Replace::new(Capture::new()), old, new);
    diff_deadline(alg, &mut d, old, old_range, new, new_range, deadline).unwrap();
    d.into_inner().into_inner().into_ops()
}

/// Creates a diff between old and new with the given algorithm capturing the ops.
pub fn capture_diff_slices<T>(alg: Algorithm, old: &[T], new: &[T]) -> Vec<DiffOp>
where
    T: Eq + Hash + Ord,
{
    capture_diff_slices_deadline(alg, old, new, None)
}

/// Creates a diff between old and new with the given algorithm capturing the ops.
///
/// Works like [`capture_diff_slices`] but with an optional deadline.
pub fn capture_diff_slices_deadline<T>(
    alg: Algorithm,
    old: &[T],
    new: &[T],
    deadline: Option<Instant>,
) -> Vec<DiffOp>
where
    T: Eq + Hash + Ord,
{
    capture_diff_deadline(alg, old, 0..old.len(), new, 0..new.len(), deadline)
}

/// Creates a diff between old and new slices using custom comparison logic.
///
/// This allows diffing elements by a subset of their fields or with custom
/// equality semantics without needing wrapper types. The `hash` and `eq`
/// closures must be consistent: if `eq(a, b)` returns true, then `a` and `b`
/// must produce the same hash.
pub fn capture_diff_slices_by<T>(
    alg: Algorithm,
    old: &[T],
    new: &[T],
    hash: impl Fn(&T, &mut dyn Hasher),
    eq: impl Fn(&T, &T) -> bool,
) -> Vec<DiffOp> {
    capture_diff_slices_by_deadline(alg, old, new, hash, eq, None)
}

/// Creates a diff between old and new slices using custom comparison logic.
///
/// Works like [`capture_diff_slices_by`] but with an optional deadline.
pub fn capture_diff_slices_by_deadline<T>(
    alg: Algorithm,
    old: &[T],
    new: &[T],
    hash: impl Fn(&T, &mut dyn Hasher),
    eq: impl Fn(&T, &T) -> bool,
    deadline: Option<Instant>,
) -> Vec<DiffOp> {
    let id = IdentifyDistinct::<u32>::new_by(old, new, &hash, &eq);
    let mut d = Compact::new(
        Replace::new(Capture::new()),
        id.old_lookup(),
        id.new_lookup(),
    );
    diff_deadline(
        alg,
        &mut d,
        id.old_lookup(),
        id.old_range(),
        id.new_lookup(),
        id.new_range(),
        deadline,
    )
    .unwrap();
    d.into_inner().into_inner().into_ops()
}

/// Creates a diff between old and new slices using a key extractor.
///
/// Elements are compared by the key returned from the `key` closure.
/// This is a convenience wrapper around [`capture_diff_slices_by`].
pub fn capture_diff_slices_by_key<T, K: Hash + Eq>(
    alg: Algorithm,
    old: &[T],
    new: &[T],
    key: impl Fn(&T) -> K,
) -> Vec<DiffOp> {
    capture_diff_slices_by_key_deadline(alg, old, new, key, None)
}

/// Creates a diff between old and new slices using a key extractor.
///
/// Works like [`capture_diff_slices_by_key`] but with an optional deadline.
pub fn capture_diff_slices_by_key_deadline<T, K: Hash + Eq>(
    alg: Algorithm,
    old: &[T],
    new: &[T],
    key: impl Fn(&T) -> K,
    deadline: Option<Instant>,
) -> Vec<DiffOp> {
    capture_diff_slices_by_deadline(
        alg,
        old,
        new,
        |item, mut state| key(item).hash(&mut state),
        |a, b| key(a) == key(b),
        deadline,
    )
}

/// Creates a diff between old and new f32 slices with epsilon comparison.
pub fn capture_diff_slices_fp(
    alg: Algorithm,
    old: &[f32],
    new: &[f32],
    epsilon: f32,
) -> Vec<DiffOp> {
    capture_diff_slices_fp_deadline(alg, old, new, epsilon, None)
}

/// Creates a diff between old and new f64 slices with epsilon comparison.
pub fn capture_diff_slices_fp_f64(
    alg: Algorithm,
    old: &[f64],
    new: &[f64],
    epsilon: f64,
) -> Vec<DiffOp> {
    capture_diff_slices_fp_f64_deadline(alg, old, new, epsilon, None)
}

/// Creates a diff between old and new f32 slices with epsilon comparison and deadline.
pub fn capture_diff_slices_fp_deadline(
    alg: Algorithm,
    old: &[f32],
    new: &[f32],
    epsilon: f32,
    deadline: Option<Instant>,
) -> Vec<DiffOp> {
    capture_diff_fp_deadline(alg, old, 0..old.len(), new, 0..new.len(), epsilon, deadline)
}

/// Creates a diff between old and new f64 slices with epsilon comparison and deadline.
pub fn capture_diff_slices_fp_f64_deadline(
    alg: Algorithm,
    old: &[f64],
    new: &[f64],
    epsilon: f64,
    deadline: Option<Instant>,
) -> Vec<DiffOp> {
    capture_diff_fp_f64_deadline(alg, old, 0..old.len(), new, 0..new.len(), epsilon, deadline)
}

fn capture_diff_fp_deadline(
    alg: Algorithm,
    old: &[f32],
    old_range: Range<usize>,
    new: &[f32],
    new_range: Range<usize>,
    epsilon: f32,
    deadline: Option<Instant>,
) -> Vec<DiffOp> {
    let mut d = Compact::new(Replace::new(Capture::new()), old, new);
    let result = match alg {
        Algorithm::Myers => crate::algorithms::myers::diff_fp_deadline(
            &mut d, old, old_range, new, new_range, epsilon, deadline,
        ),
        Algorithm::Patience => crate::algorithms::patience::diff_fp_deadline(
            &mut d, old, old_range, new, new_range, epsilon, deadline,
        ),
        Algorithm::Lcs => crate::algorithms::lcs::diff_fp_deadline(
            &mut d, old, old_range, new, new_range, epsilon, deadline,
        ),
    };
    result.unwrap();
    d.into_inner().into_inner().into_ops()
}

fn capture_diff_fp_f64_deadline(
    alg: Algorithm,
    old: &[f64],
    old_range: Range<usize>,
    new: &[f64],
    new_range: Range<usize>,
    epsilon: f64,
    deadline: Option<Instant>,
) -> Vec<DiffOp> {
    let mut d = Compact::new(Replace::new(Capture::new()), old, new);
    let result = match alg {
        Algorithm::Myers => crate::algorithms::myers::diff_fp_f64_deadline(
            &mut d, old, old_range, new, new_range, epsilon, deadline,
        ),
        Algorithm::Patience => crate::algorithms::patience::diff_fp_f64_deadline(
            &mut d, old, old_range, new, new_range, epsilon, deadline,
        ),
        Algorithm::Lcs => crate::algorithms::lcs::diff_fp_f64_deadline(
            &mut d, old, old_range, new, new_range, epsilon, deadline,
        ),
    };
    result.unwrap();
    d.into_inner().into_inner().into_ops()
}

/// Return a measure of similarity in the range `0..=1`.
///
/// A ratio of `1.0` means the two sequences are a complete match, a
/// ratio of `0.0` would indicate completely distinct sequences.  The input
/// is the sequence of diff operations and the length of the old and new
/// sequence.
pub fn get_diff_ratio(ops: &[DiffOp], old_len: usize, new_len: usize) -> f32 {
    let matches = ops
        .iter()
        .map(|op| {
            if let DiffOp::Equal { len, .. } = *op {
                len
            } else {
                0
            }
        })
        .sum::<usize>();
    let len = old_len + new_len;
    if len == 0 {
        1.0
    } else {
        2.0 * matches as f32 / len as f32
    }
}

/// Isolate change clusters by eliminating ranges with no changes.
///
/// This will leave holes behind in long periods of equal ranges so that
/// you can build things like unified diffs.
pub fn group_diff_ops(mut ops: Vec<DiffOp>, n: usize) -> Vec<Vec<DiffOp>> {
    if ops.is_empty() {
        return vec![];
    }

    let mut pending_group = Vec::new();
    let mut rv = Vec::new();

    if let Some(DiffOp::Equal {
        old_index,
        new_index,
        len,
    }) = ops.first_mut()
    {
        let offset = (*len).saturating_sub(n);
        *old_index += offset;
        *new_index += offset;
        *len -= offset;
    }

    if let Some(DiffOp::Equal { len, .. }) = ops.last_mut() {
        *len -= (*len).saturating_sub(n);
    }

    for op in ops.into_iter() {
        if let DiffOp::Equal {
            old_index,
            new_index,
            len,
        } = op
        {
            // End the current group and start a new one whenever
            // there is a large range with no changes.
            if len > n * 2 {
                pending_group.push(DiffOp::Equal {
                    old_index,
                    new_index,
                    len: n,
                });
                rv.push(pending_group);
                let offset = len.saturating_sub(n);
                pending_group = vec![DiffOp::Equal {
                    old_index: old_index + offset,
                    new_index: new_index + offset,
                    len: len - offset,
                }];
                continue;
            }
        }
        pending_group.push(op);
    }

    match &pending_group[..] {
        &[] | &[DiffOp::Equal { .. }] => {}
        _ => rv.push(pending_group),
    }

    rv
}

#[test]
fn test_non_string_iter_change() {
    use crate::ChangeTag;

    let old = vec![1, 2, 3];
    let new = vec![1, 2, 4];
    let ops = capture_diff_slices(Algorithm::Myers, &old, &new);
    let changes: Vec<_> = ops
        .iter()
        .flat_map(|x| x.iter_changes(&old, &new))
        .map(|x| (x.tag(), x.value()))
        .collect();

    assert_eq!(
        changes,
        vec![
            (ChangeTag::Equal, 1),
            (ChangeTag::Equal, 2),
            (ChangeTag::Delete, 3),
            (ChangeTag::Insert, 4),
        ]
    );
}

#[test]
fn test_issue_78_delete_new_index() {
    let a = vec![1, 1, 0, 2];
    let b = vec![2, 1, 0, 0, 0, 0];
    let ops = capture_diff_slices(Algorithm::Myers, &a, &b);

    for i in 1..ops.len() {
        let prev = &ops[i - 1];
        let op = &ops[i];
        assert_eq!(
            op.old_range().start,
            prev.old_range().end,
            "old_index gap at op {i}: {prev:?} -> {op:?}"
        );
        assert_eq!(
            op.new_range().start,
            prev.new_range().end,
            "new_index gap at op {i}: {prev:?} -> {op:?}"
        );
    }
}

#[test]
fn test_capture_diff_slices_by_key() {
    #[derive(Debug, Clone)]
    struct Record {
        id: u32,
        _value: &'static str,
    }

    let old = vec![
        Record { id: 1, _value: "a" },
        Record { id: 2, _value: "b" },
        Record { id: 3, _value: "c" },
    ];
    let new = vec![
        Record { id: 1, _value: "x" },
        Record { id: 4, _value: "d" },
        Record { id: 3, _value: "z" },
    ];

    let ops = capture_diff_slices_by_key(Algorithm::Myers, &old, &new, |r| r.id);
    let changes: Vec<_> = ops
        .iter()
        .flat_map(|op| op.iter_changes(&old, &new))
        .map(|c| (c.tag(), c.value().id))
        .collect();

    use crate::ChangeTag;
    assert_eq!(
        changes,
        vec![
            (ChangeTag::Equal, 1),
            (ChangeTag::Delete, 2),
            (ChangeTag::Insert, 4),
            (ChangeTag::Equal, 3),
        ]
    );
}

#[test]
fn test_capture_diff_slices_by() {
    let old = vec!["HELLO", "world", "FOO"];
    let new = vec!["hello", "WORLD", "foo"];

    let ops = capture_diff_slices_by(
        Algorithm::Myers,
        &old,
        &new,
        |s, state| {
            for b in s.to_ascii_lowercase().bytes() {
                Hasher::write_u8(state, b);
            }
        },
        |a, b| a.eq_ignore_ascii_case(b),
    );

    assert_eq!(ops.len(), 1);
    assert!(matches!(ops[0], DiffOp::Equal { len: 3, .. }));
}

#[test]
fn test_capture_diff_slices_by_key_all_algorithms() {
    let old = vec![(1, "a"), (2, "b"), (3, "c")];
    let new = vec![(1, "x"), (3, "y"), (4, "z")];

    for alg in [Algorithm::Myers, Algorithm::Patience, Algorithm::Lcs] {
        let ops = capture_diff_slices_by_key(alg, &old, &new, |x| x.0);
        let changes: Vec<_> = ops
            .iter()
            .flat_map(|op| op.iter_changes(&old, &new))
            .map(|c| (c.tag(), c.value().0))
            .collect();

        use crate::ChangeTag;
        assert_eq!(
            changes,
            vec![
                (ChangeTag::Equal, 1),
                (ChangeTag::Delete, 2),
                (ChangeTag::Equal, 3),
                (ChangeTag::Insert, 4),
            ],
            "failed for {alg:?}"
        );
    }
}

#[test]
fn test_capture_diff_slices_by_key_empty() {
    let empty: Vec<(i32, i32)> = vec![];
    let non_empty = vec![(1, 2)];

    let ops = capture_diff_slices_by_key(Algorithm::Myers, &empty, &empty, |x| x.0);
    assert!(ops.is_empty());

    let ops = capture_diff_slices_by_key(Algorithm::Myers, &empty, &non_empty, |x| x.0);
    assert_eq!(ops.len(), 1);
    assert!(matches!(ops[0], DiffOp::Insert { new_len: 1, .. }));

    let ops = capture_diff_slices_by_key(Algorithm::Myers, &non_empty, &empty, |x| x.0);
    assert_eq!(ops.len(), 1);
    assert!(matches!(ops[0], DiffOp::Delete { old_len: 1, .. }));
}

#[test]
fn test_capture_diff_slices_by_key_all_equal() {
    let old = vec![(1, "a"), (2, "b")];
    let new = vec![(1, "x"), (2, "y")];

    let ops = capture_diff_slices_by_key(Algorithm::Myers, &old, &new, |x| x.0);
    assert_eq!(ops.len(), 1);
    assert!(matches!(ops[0], DiffOp::Equal { len: 2, .. }));
}

#[test]
fn test_capture_diff_slices_by_key_all_different() {
    let old = vec![(1,), (2,)];
    let new = vec![(3,), (4,)];

    let ops = capture_diff_slices_by_key(Algorithm::Myers, &old, &new, |x| x.0);
    let has_replace = ops.iter().any(|op| matches!(op, DiffOp::Replace { .. }));
    assert!(has_replace);
}

#[test]
fn test_fp_epsilon() {
    let old = vec![1.0, 2.0, 3.0];
    let new = vec![1.001, 2.0, 2.999];

    let ops_tight = capture_diff_slices_fp(Algorithm::Myers, &old, &new, 0.0001);
    assert!(ops_tight.len() > 1);

    let ops_loose = capture_diff_slices_fp(Algorithm::Myers, &old, &new, 0.01);
    assert_eq!(ops_loose.len(), 1);
}

#[test]
fn test_fp_nan() {
    let old = vec![f32::NAN];
    let new = vec![f32::NAN];

    let ops = capture_diff_slices_fp(Algorithm::Myers, &old, &new, 0.001);
    assert_eq!(ops.len(), 1);
}
