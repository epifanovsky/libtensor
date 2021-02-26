#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tof_set.h>
#include "tof_set_test.h"

namespace libtensor {


void tof_set_test::perform() throw(libtest::test_exception) {

    test_1(1.0);
    test_2(-2.0);
    test_3(5.0);
    test_4(-0.3);
}


void tof_set_test::test_1(float d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tof_set_test::test_1(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<float> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = 10;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<1, float, allocator_t> t1(dims), t2(dims);

        tof_set<1>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<1, float> c1(t1);
            const float *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tof_set<1>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<1, float> c2(t2);
            const float *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tof_set_test::test_2(float d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tof_set_test::test_2(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<float> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 3;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<2, float, allocator_t> t1(dims), t2(dims);

        tof_set<2>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<2, float> c1(t1);
            const float *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tof_set<2>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<2, float> c2(t2);
            const float *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tof_set_test::test_3(float d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tof_set_test::test_3(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<float> allocator_t;

    try {

        index<3> i1, i2;
        i2[0] = 10; i2[1] = 2; i2[2] = 21;
        dimensions<3> dims(index_range<3>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<3, float, allocator_t> t1(dims), t2(dims);

        tof_set<3>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<3, float> c1(t1);
            const float *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tof_set<3>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<3, float> c2(t2);
            const float *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tof_set_test::test_4(float d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tof_set_test::test_4(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<float> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 1; i2[1] = 4; i2[2] = 9; i2[3] = 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<4, float, allocator_t> t1(dims), t2(dims);

        tof_set<4>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<4, float> c1(t1);
            const float *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        tof_set<4>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<4, float> c2(t2);
            const float *p = c2.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != d) {
                std::ostringstream ss;
                ss << "Bad value at t2[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c2.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

