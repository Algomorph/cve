#define PY_ARRAY_UNIQUE_SYMBOL cve_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <imgproc/imgproc.hpp>
#include <bgsegm/bgsegm.hpp>

namespace cve {

    using namespace boost::python;

/**
 * Example function. Basic inner matrix product using explicit matrix conversion.
 * @param left left-hand matrix operand (NdArray required)
 * @param right right-hand matrix operand (NdArray required)
 * @return an NdArray representing the dot-product of the left and right operands
 */
    PyObject *dot(PyObject *left, PyObject *right) {

        cv::Mat leftMat, rightMat;
        leftMat = cve::fromNDArrayToMat(left);
        rightMat = cve::fromNDArrayToMat(right);
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        // Check that the 2-D matrices can be legally multiplied.
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;
        PyObject * ret = cve::fromMatToNDArray(result);
        return ret;
    }



//This example uses Mat directly, but we won't need to worry about the conversion
/**
 * Example function. Basic inner matrix product using implicit matrix conversion.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;

        return result;
    }

#if (PY_VERSION_HEX >= 0x03000000)

        static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (cve) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                cve::matToNDArrayBoostConverter>();
        cve::matFromNDArrayBoostConverter();

        //expose module-level functions
        def("dot", dot);
        def("dot2", dot2);
        def("non_max_suppression", cve::non_max_suppression);
        class_<BackgroundSubtractorIMBS>("BackgroundSubtractorIMBS", "Independent Multimodal Background Subtraction, D. Bloisi and L. Locchi, 2012",
                                         init<optional<double, unsigned int,unsigned int,
                                                 double, unsigned int, unsigned int,
                                                 double, double, double, double,
                                                 double, double,
                                                 bool, bool>>(args("fps","fg_threshold", "association_threshold",
                                                             "sampling_interval", "min_bin_height", "num_samples",
                                                             "alpha", "beta", "tau_s", "tau_h",
                                                             "min_area", "persistence_period",
                                                             "use_morphological_filtering", "rebuild_model_continously"))
        ).def("apply", &BackgroundSubtractorIMBS::apply_pythonic);
    }

} //end namespace cve
