#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "../test_utils.h"

using namespace libtensor;


int test_1(double d) {

    std::ostringstream ss;
    ss << "tod_set_test::test_1(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<1> i1, i2;
        i2[0] = 10;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<1, double, allocator_t> t1(dims), t2(dims);

        tod_set<1>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<1, double> c1(t1);
            const double *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tod_set<1>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<1, double> c2(t2);
            const double *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_2(double d) {

    std::ostringstream ss;
    ss << "tod_set_test::test_2(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<2> i1, i2;
        i2[0] = 10; i2[1] = 3;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<2, double, allocator_t> t1(dims), t2(dims);

        tod_set<2>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<2, double> c1(t1);
            const double *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tod_set<2>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<2, double> c2(t2);
            const double *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_3(double d) {

    std::ostringstream ss;
    ss << "tod_set_test::test_3(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<3> i1, i2;
        i2[0] = 10; i2[1] = 2; i2[2] = 21;
        dimensions<3> dims(index_range<3>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<3, double, allocator_t> t1(dims), t2(dims);

        tod_set<3>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<3, double> c1(t1);
            const double *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tod_set<3>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<3, double> c2(t2);
            const double *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_4(double d) {

    std::ostringstream ss;
    ss << "tod_set_test::test_4(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<4> i1, i2;
        i2[0] = 1; i2[1] = 4; i2[2] = 9; i2[3] = 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<4, double, allocator_t> t1(dims), t2(dims);

        tod_set<4>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<4, double> c1(t1);
            const double *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tod_set<4>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<4, double> c2(t2);
            const double *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                return fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        return fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_1(1.0) |
    test_2(-2.0) |
    test_3(5.0) |
    test_4(-0.3) |

    0;
}


