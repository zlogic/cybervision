// exact C++ implementation of lowe's sift program
// Copyright (C) zerofrog(@gmail.com), 2008-2009
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//Lesser GNU General Public License for more details.
//
//You should have received a copy of the GNU Lesser General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.
#include <vector>
#include <boost/shared_ptr.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle
#include <boost/python.hpp>
#include <boost/python/exception_translator.hpp>
#include <pyconfig.h>
#include <numpy/arrayobject.h>

#include "siftfast.h"

#define CHECK_POINTER(p) { \
    if( (p) == NULL ) throw siftfast_exception("invalid pointer"); \
}

using namespace boost::python;
using namespace std;

extern int DoubleImSize;
extern int Scales;
extern float InitSigma;
extern float PeakThresh;

struct siftfast_exception : std::exception
{
    siftfast_exception() : std::exception(), _s("unknown exception") {}
    siftfast_exception(const string& s) : std::exception() { _s = "siftfast: " + s; }
    virtual ~siftfast_exception() throw() {}
    char const* what() const throw() { return _s.c_str(); }
    string _s;
};

void translate_siftfast_exception(siftfast_exception const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

inline object toPyArrayN(const float* pvalues, int N)
{
    npy_intp dims[] = {N};
    PyObject *pyvalues = PyArray_SimpleNew(1,dims, PyArray_FLOAT);
    if( pvalues != NULL )
        memcpy(PyArray_DATA(pyvalues),pvalues,N*sizeof(float));
    return static_cast<numeric::array>(handle<>(pyvalues));
}

template <typename T>
inline vector<T> ExtractArray(object o)
{
    vector<T> v(len(o));
    for(size_t i = 0; i < v.size(); ++i)
        v[i] = extract<T>(o[i]);
    return v;
}

inline vector<float> ExtractFloatArray(object oraw)
{
    object o = oraw.attr("flat");

    // check the types of o
    extract<float> xr(o[0]);
    if( xr.check() )
        return ExtractArray<float>(o);

    vector<float> v(len(o));
    object onew = ((numeric::array)oraw).astype("f8").attr("flat");
    for(size_t i = 0; i < v.size(); ++i)
        v[i] = (float)(extract<double>(onew[i]));
    return v;
}

class PyImage
{
public:
    PyImage(int width, int height) : width(width), height(height)
    {
        if( width <= 0 || height <= 0 )
            throw siftfast_exception("bad image dimensions");
        stride = (width+3)&~3;
        vimage.resize(width*stride);
    }
    PyImage(int width, int height, int stride, object arr) : width(width), height(height), stride(stride)
    {
        if( width <= 0 || height <= 0 )
            throw siftfast_exception("bad image dimensions");
        vimage = ExtractFloatArray(arr);
        if( (int)vimage.size() != stride*height )
            throw siftfast_exception("data array not correct size");
    }

    void SetData(object arr)
    {
        object shape = arr.attr("shape");
        if( len(shape) != 2 )
            throw siftfast_exception("array needs 2 dimensions");
        if( height != extract<int>(shape[0]) )
            throw siftfast_exception("array rows do not match height");
        if( width != extract<int>(shape[1]) )
            throw siftfast_exception("array columns do not match width");

        string dtype = extract<string>(arr.attr("dtype").attr("name"));
        if( (dtype.size() >= 3 && dtype[0] == 'i' && dtype[1] == 'n' && dtype[2] == 't') ||
            (dtype.size() >= 4 && dtype[1] == 'i' && dtype[2] == 'n' && dtype[3] == 't') ) {
            extract<int> xi(arr[0][0]);
            if( xi.check() ) {
                for(int i = 0; i < height; ++i)
                    for(int j = 0; j < width; ++j)
                        vimage[i*stride+j] = extract<int>(arr[i][j])*(1.0f/255.0f);
                return;
            }
        }

        extract<float> xr(arr[0][0]);
        if( xr.check() ) {
            for(int i = 0; i < height; ++i)
                for(int j = 0; j < width; ++j)
                    vimage[i*stride+j] = extract<float>(arr[i][j]);
            return;
        }
        extract<double> xd(arr[0][0]);
        if( xd.check() ) {
            for(int i = 0; i < height; ++i)
                for(int j = 0; j < width; ++j)
                    vimage[i*stride+j] = extract<double>(arr[i][j]);
            return;
        }

        throw siftfast_exception("array not in correct format");
//        printf("heree!\n");
//        object arrnew = ((numeric::array)oraw).astype("f8");
//        extract<double> arrnew(arr[0]);
//        if( xr.check() ) {
//            for(int i = 0; i < height; ++i)
//                for(int j = 0; j < width; ++j)
//                    vimage[i*stride+j] = extract<double>(arrnew[i][j]);
//        }
    }

    int width,height,stride;
    vector<float> vimage;
};

class Image_pickle_suite : public pickle_suite
{
public:
    static tuple getinitargs(const PyImage& im)
    {
        return make_tuple(im.width,im.height,im.stride,toPyArrayN(&im.vimage[0],im.vimage.size()));
    }
};

object PyGetKeypoints(PyImage& im)
{
    struct ImageSt siftimage;
    siftimage.rows = im.height;
    siftimage.cols = im.width;
    siftimage.pixels = &im.vimage[0];
    siftimage.stride = im.stride;
    
    Keypoint keypts = GetKeypoints(&siftimage);

    int numkeys = 0;
    Keypoint key = keypts;
    while(key) {
        numkeys++;
        key = key->next;
    }

    npy_intp dims[2] = {numkeys,4};
    PyObject *pyframes = PyArray_SimpleNew(2,dims, PyArray_FLOAT);
    float* pframes = (float*)PyArray_DATA(pyframes);
    
    dims[1] = 128;
    PyObject *pydesc = PyArray_SimpleNew(2,dims, PyArray_FLOAT);
    float* pdesc = (float*)PyArray_DATA(pydesc);

    int index = 0;
    key = keypts;
    while(key) {

        for(int j = 0; j < 128; ++j)
            pdesc[128*index+j] = key->descrip[j];

        pframes[4*index+0] = key->col;
        pframes[4*index+1] = key->row;
        pframes[4*index+2] = key->ori;
        pframes[4*index+3] = key->scale;

        key = key->next;
        ++index;
    }

    FreeKeypoints(keypts);
    DestroyAllImages();

    return make_tuple(static_cast<numeric::array>(handle<>(pyframes)),static_cast<numeric::array>(handle<>(pydesc)));
}

struct DummyStruct {};

BOOST_PYTHON_MODULE(siftfastpy)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
    register_exception_translator<siftfast_exception>(&translate_siftfast_exception);

    def("DestroyAllResources",DestroyAllResources);
    def("GetKeypoints",PyGetKeypoints);

    class_<PyImage>("Image", no_init)
        .def(init<int,int>())
        .def(init<int,int,int,object>())
        .def_readonly("width",&PyImage::width)
        .def_readonly("height",&PyImage::height)
        .def("SetData",&PyImage::SetData)
        .def_pickle(Image_pickle_suite())
        ;

    {
        scope options = class_<DummyStruct>("options")
            .def_readwrite("DoubleImSize",&DoubleImSize)
            .def_readwrite("Scales",&Scales)
            .def_readwrite("InitSigma",&InitSigma)
            .def_readwrite("PeakThresh",&PeakThresh);
    }
}
