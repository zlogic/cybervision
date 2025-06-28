use std::{ops::Range, slice};

use rayon::{
    iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge},
    prelude::{IndexedParallelIterator, ParallelIterator},
};

#[derive(Copy, Clone, Debug)]
pub struct Point2D<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point2D<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Point2D<T> {
        Point2D { x, y }
    }
}

#[derive(Clone)]
pub struct Grid<T>
where
    T: Send + Sync,
{
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<'a, T: Clone> Grid<T>
where
    T: Send + Sync,
{
    pub fn new(width: usize, height: usize, default_value: T) -> Grid<T> {
        let data = vec![default_value; width * height];
        Grid {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn val(&self, x: usize, y: usize) -> &T {
        assert!(x < self.width && y < self.height, "Index out of bounds");
        &self.data[self.width * y + x]
    }

    #[inline]
    pub fn val_mut(&mut self, x: usize, y: usize) -> &mut T {
        assert!(x < self.width && y < self.height, "Index out of bounds");
        &mut self.data[self.width * y + x]
    }

    pub fn iter(&'a self) -> GridIter<'a, T> {
        let range = 0..self.data.len();
        GridIter {
            iter: self.data.iter(),
            range,
            width: self.width,
        }
    }

    pub fn par_iter(&'a self) -> ParGridIter<'a, T> {
        let range = 0..self.data.len();
        ParGridIter {
            data: self.data.as_slice(),
            range,
            width: self.width,
        }
    }

    pub fn iter_mut(&'a mut self) -> GridIterMut<'a, T> {
        let range = 0..self.data.len();
        GridIterMut {
            iter: self.data.iter_mut(),
            range,
            width: self.width,
        }
    }

    pub fn par_iter_mut(&'a mut self) -> ParGridIterMut<'a, T> {
        let range = 0..self.data.len();
        ParGridIterMut {
            data: self.data.as_mut_slice(),
            range,
            width: self.width,
        }
    }
}

type GridIter<'a, T> = GridIterImpl<slice::Iter<'a, T>>;
type GridIterMut<'a, T> = GridIterImpl<slice::IterMut<'a, T>>;

pub struct GridIterImpl<I>
where
    I: Iterator + ExactSizeIterator + DoubleEndedIterator,
{
    iter: I,
    range: Range<usize>,
    width: usize,
}

impl<I> Iterator for GridIterImpl<I>
where
    I: Iterator + ExactSizeIterator + DoubleEndedIterator,
{
    type Item = (usize, usize, I::Item);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let val = self.iter.next()?;
        let i = self.range.start;
        self.range.start += 1;
        let x = i % self.width;
        let y = i / self.width;
        Some((x, y, val))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }
}

unsafe impl<I> Send for GridIterImpl<I> where I: Iterator + ExactSizeIterator + DoubleEndedIterator {}

impl<I> ExactSizeIterator for GridIterImpl<I> where
    I: Iterator + ExactSizeIterator + DoubleEndedIterator
{
}

impl<I> DoubleEndedIterator for GridIterImpl<I>
where
    I: Iterator + ExactSizeIterator + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let val = self.iter.next_back()?;
        self.range.end -= 1;
        let i = self.range.end;
        let x = i % self.width;
        let y = i / self.width;
        Some((x, y, val))
    }
}

pub struct GridProducer<'a, T>(ParGridIter<'a, T>)
where
    T: Send + Sync;

pub struct GridProducerMut<'a, T>(ParGridIterMut<'a, T>)
where
    T: Send + Sync;

pub struct ParGridIter<'a, T>
where
    T: Send + Sync,
{
    data: &'a [T],
    range: Range<usize>,
    width: usize,
}

pub struct ParGridIterMut<'a, T>
where
    T: Send + Sync,
{
    data: &'a mut [T],
    range: Range<usize>,
    width: usize,
}

impl<'a, T> Producer for GridProducer<'a, T>
where
    T: Send + Sync,
{
    type Item = <GridIterImpl<slice::Iter<'a, T>> as Iterator>::Item;

    type IntoIter = GridIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            iter: self.0.data.iter(),
            range: self.0.range,
            width: self.0.width,
        }
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let (start, end) = (self.0.range.start, self.0.range.end);
        let mid = (start + index).min(end);
        let left_range = start..mid;
        let right_range = mid..end;
        let (left_data, right_data) = self.0.data.split_at(index);
        let left = GridProducer(ParGridIter {
            data: left_data,
            range: left_range,
            width: self.0.width,
        });
        let right = GridProducer(ParGridIter {
            data: right_data,
            range: right_range,
            width: self.0.width,
        });
        (left, right)
    }
}

impl<'a, T> Producer for GridProducerMut<'a, T>
where
    T: Send + Sync,
{
    type Item = <GridIterImpl<slice::IterMut<'a, T>> as Iterator>::Item;

    type IntoIter = GridIterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            iter: self.0.data.iter_mut(),
            range: self.0.range,
            width: self.0.width,
        }
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let (start, end) = (self.0.range.start, self.0.range.end);
        let mid = (start + index).min(end);
        let left_range = start..mid;
        let right_range = mid..end;
        let (left_data, right_data) = self.0.data.split_at_mut(index);
        let left = GridProducerMut(ParGridIterMut {
            data: left_data,
            range: left_range,
            width: self.0.width,
        });
        let right = GridProducerMut(ParGridIterMut {
            data: right_data,
            range: right_range,
            width: self.0.width,
        });
        (left, right)
    }
}

impl<'a, T> IndexedParallelIterator for ParGridIter<'a, T>
where
    T: Send + Sync,
{
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = GridProducer(self);
        callback.callback(producer)
    }
}

impl<'a, T> IndexedParallelIterator for ParGridIterMut<'a, T>
where
    T: Send + Sync,
{
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = GridProducerMut(self);
        callback.callback(producer)
    }
}

impl<'a, T> ParallelIterator for ParGridIter<'a, T>
where
    T: Send + Sync,
{
    type Item = <GridIterImpl<slice::Iter<'a, T>> as Iterator>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.range.len())
    }
}

impl<'a, T> ParallelIterator for ParGridIterMut<'a, T>
where
    T: Send + Sync,
{
    type Item = <GridIterImpl<slice::IterMut<'a, T>> as Iterator>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.range.len())
    }
}
