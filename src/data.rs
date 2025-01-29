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

    pub fn iter(&'a self) -> GridIterConst<'a, T> {
        let view = GridStorageConst {
            grid: self,
            phantom: Default::default(),
            width: self.width,
        };
        GridIter {
            grid: view,
            phantom: Default::default(),
            range: (0..self.data.len()),
        }
    }

    pub fn par_iter(&'a self) -> ParGridIterConst<'a, T> {
        ParGridIter { it: self.iter() }
    }

    pub fn iter_mut(&'a mut self) -> GridIterMut<'a, T> {
        let view = GridStorageMut {
            grid: self,
            phantom: Default::default(),
            width: self.width,
        };
        GridIter {
            grid: view,
            phantom: Default::default(),
            range: (0..self.data.len()),
        }
    }

    pub fn par_iter_mut(&'a mut self) -> ParGridIterMut<'a, T> {
        ParGridIter {
            it: self.iter_mut(),
        }
    }
}

type GridIterConst<'a, T> = GridIter<'a, T, GridStorageConst<'a, T>, (usize, usize, &'a T)>;
type GridIterMut<'a, T> = GridIter<'a, T, GridStorageMut<'a, T>, (usize, usize, &'a mut T)>;
type ParGridIterConst<'a, T> = ParGridIter<'a, T, GridStorageConst<'a, T>, (usize, usize, &'a T)>;
type ParGridIterMut<'a, T> = ParGridIter<'a, T, GridStorageMut<'a, T>, (usize, usize, &'a mut T)>;

pub trait GridView<'a, T> {
    type Item;
    unsafe fn get(&self, i: usize) -> Self::Item;
}

pub struct GridStorage<'a, T, G>
where
    T: Send + Sync,
{
    grid: G,
    phantom: PhantomData<&'a Grid<T>>,
    width: usize,
}

type GridStorageConst<'a, T> = GridStorage<'a, T, *const Grid<T>>;
type GridStorageMut<'a, T> = GridStorage<'a, T, *mut Grid<T>>;

impl<'a, T> GridView<'a, T> for GridStorageConst<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a T);

    unsafe fn get(&self, i: usize) -> Self::Item {
        let width = self.width;
        let x = i % width;
        let y = i / width;
        let val = &(*self.grid).data[i];
        (x, y, val)
    }
}

impl<'a, T> GridView<'a, T> for GridStorageMut<'a, T>
where
    T: Send + Sync,
{
    type Item = (usize, usize, &'a mut T);

    unsafe fn get(&self, i: usize) -> Self::Item {
        let width = self.width;
        let x = i % width;
        let y = i / width;
        let val = &mut (*self.grid).data[i];
        (x, y, val)
    }
}

impl<T> Clone for GridStorageConst<'_, T>
where
    T: Send + Sync,
{
    fn clone(&self) -> Self {
        GridStorageConst {
            grid: self.grid,
            phantom: Default::default(),
            width: self.width,
        }
    }
}

impl<T> Clone for GridStorageMut<'_, T>
where
    T: Send + Sync,
{
    fn clone(&self) -> Self {
        GridStorageMut {
            grid: self.grid,
            phantom: Default::default(),
            width: self.width,
        }
    }
}

pub struct GridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync,
{
    grid: G,
    phantom: PhantomData<&'a Grid<T>>,
    range: Range<usize>,
}

impl<'a, T, G, I> Iterator for GridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync,
{
    type Item = G::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.range.next()?;
        let val = unsafe { self.grid.get(i) };
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.range.len();
        (hint, Some(hint))
    }

    #[inline]
    fn count(self) -> usize {
        self.range.len()
    }
}

unsafe impl<'a, T, G, I> Send for GridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync,
{
}

impl<'a, T, G, I> ExactSizeIterator for GridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync,
{
}

impl<'a, T, G, I> DoubleEndedIterator for GridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.range.next_back()?;
        let val = unsafe { self.grid.get(i) };
        Some(val)
    }
}

struct GridProducer<'a, T, G, I>(ParGridIter<'a, T, G, I>)
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync;

pub struct ParGridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I>,
    I: Send + Sync,
{
    it: GridIter<'a, T, G, I>,
}

impl<'a, T, G, I> Producer for GridProducer<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I> + Clone,
    I: Send + Sync,
{
    type Item = G::Item;

    type IntoIter = GridIter<'a, T, G, I>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.it
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = (self.0.it.range.start + index).min(self.0.it.range.end);
        let left_range = self.0.it.range.start..mid;
        let right_range = mid..self.0.it.range.end;
        let left = ParGridIter {
            it: GridIter {
                grid: self.0.it.grid.clone(),
                phantom: Default::default(),
                range: left_range,
            },
        };
        let right = ParGridIter {
            it: GridIter {
                grid: self.0.it.grid.clone(),
                phantom: Default::default(),
                range: right_range,
            },
        };
        (Self(left), Self(right))
    }
}

impl<'a, T, G, I> IndexedParallelIterator for ParGridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I> + Clone,
    I: Send + Sync,
{
    #[inline]
    fn len(&self) -> usize {
        self.it.range.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = GridProducer(self);
        callback.callback(producer)
    }
}

impl<'a, T, G, I> ParallelIterator for ParGridIter<'a, T, G, I>
where
    T: Send + Sync,
    G: GridView<'a, T, Item = I> + Clone,
    I: Send + Sync,
{
    type Item = G::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.it.range.len())
    }
}
