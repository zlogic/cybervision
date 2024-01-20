use std::{marker::PhantomData, ops::Range};

use rayon::{
    iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
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

    pub fn iter(&self) -> GridIter<'a, T> {
        GridIter {
            grid: self,
            phantom: Default::default(),
            range: (0..self.data.len()),
            width: self.width,
        }
    }

    pub fn par_iter(&self) -> ParGridIter<'a, T> {
        ParGridIter { it: self.iter() }
    }

    pub fn iter_mut(&mut self) -> GridIterMut<'a, T> {
        GridIterMut {
            grid: self,
            phantom: Default::default(),
            range: (0..self.data.len()),
            width: self.width,
        }
    }

    pub fn par_iter_mut(&mut self) -> ParGridIterMut<'a, T> {
        ParGridIterMut {
            it: self.iter_mut(),
        }
    }
}

pub struct GridIter<'a, T>
where
    T: Send + Sync,
{
    grid: *const Grid<T>,
    phantom: PhantomData<&'a Grid<T>>,
    range: Range<usize>,
    width: usize,
}

impl<'a, T> Iterator for GridIter<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.range.next()?;
        let x = i % self.width;
        let y = i / self.width;
        let val = unsafe { &(*self.grid).data[i] };
        Some((x, y, val))
    }
}

unsafe impl<'a, T> Send for GridIter<'a, T> where T: Send + Sync {}

impl<'a, T> ExactSizeIterator for GridIter<'a, T> where T: Send + Sync {}

impl<'a, T> DoubleEndedIterator for GridIter<'a, T>
where
    T: Send + Sync,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.range.next_back()?;
        let x = i % self.width;
        let y = i / self.width;
        let val = unsafe { &(*self.grid).data[i] };
        Some((x, y, val))
    }
}

pub struct ParGridIter<'a, T>
where
    T: Send + Sync,
{
    it: GridIter<'a, T>,
}

impl<'a, T> Producer for ParGridIter<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a T);

    type IntoIter = GridIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.it
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left_range = self.it.range.start..self.it.range.start + index;
        let right_range = left_range.end..self.it.range.end;
        let left = ParGridIter {
            it: GridIter {
                grid: self.it.grid,
                phantom: Default::default(),
                range: left_range,
                width: self.it.width,
            },
        };
        let right = ParGridIter {
            it: GridIter {
                grid: self.it.grid,
                phantom: Default::default(),
                range: right_range,
                width: self.it.width,
            },
        };
        (left, right)
    }
}

impl<'a, T> IndexedParallelIterator for ParGridIter<'a, T>
where
    T: Send + Sync,
{
    fn len(&self) -> usize {
        self.it.range.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

impl<'a, T> ParallelIterator for ParGridIter<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a T);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.it.range.len())
    }
}

pub struct GridIterMut<'a, T>
where
    T: Sync + Send,
{
    grid: *mut Grid<T>,
    phantom: PhantomData<&'a Grid<T>>,
    range: Range<usize>,
    width: usize,
}

impl<'a, T> Iterator for GridIterMut<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.range.next()?;
        let x = i % self.width;
        let y = i / self.width;
        let val = unsafe { &mut (*self.grid).data[i] };
        Some((x, y, val))
    }
}

unsafe impl<'a, T> Send for GridIterMut<'a, T> where T: Send + Sync {}

impl<'a, T> ExactSizeIterator for GridIterMut<'a, T> where T: Send + Sync {}

impl<'a, T> DoubleEndedIterator for GridIterMut<'a, T>
where
    T: Send + Sync,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.range.next_back()?;
        let x = i % self.width;
        let y = i / self.width;
        let val = unsafe { &mut (*self.grid).data[i] };
        Some((x, y, val))
    }
}

pub struct ParGridIterMut<'a, T>
where
    T: Send + Sync,
{
    it: GridIterMut<'a, T>,
}

impl<'a, T> Producer for ParGridIterMut<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a mut T);

    type IntoIter = GridIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.it
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left_range = self.it.range.start..self.it.range.start + index;
        let right_range = left_range.end..self.it.range.end;
        let left = ParGridIterMut {
            it: GridIterMut {
                grid: self.it.grid,
                phantom: Default::default(),
                range: left_range,
                width: self.it.width,
            },
        };
        let right = ParGridIterMut {
            it: GridIterMut {
                grid: self.it.grid,
                phantom: Default::default(),
                range: right_range,
                width: self.it.width,
            },
        };
        (left, right)
    }
}

impl<'a, T> IndexedParallelIterator for ParGridIterMut<'a, T>
where
    T: Sync + Send,
{
    fn len(&self) -> usize {
        self.it.range.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

impl<'a, T> ParallelIterator for ParGridIterMut<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a mut T);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.it.range.len())
    }
}
