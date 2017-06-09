#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_set.h>
#include "to_set_test.h"

namespace libtensor {


void to_set_test::perform() throw(libtest::test_exception) {

    test_1<double>((double)1.0);
    test_2<double>((double)-2.0);
    test_3<double>((double)5.0);
    test_4<double>((double)-0.3);

    test_1<float>((float)1.0);
    test_2<float>((float)-2.0);
    test_3<float>((float)5.0);
    test_4<float>((float)-0.3);
}


template<typename T>
void to_set_test::test_1(T d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "to_set_test::test_1(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<T> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = 10;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<1, T, allocator_t> t1(dims), t2(dims);

        to_set<1,T>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<1, T> c1(t1);
            const T *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        to_set<1,T>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<1, T> c2(t2);
            const T *p = c2.req_const_dataptr();
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


template<typename T>
void to_set_test::test_2(T d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "to_set_test::test_2(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<T> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 3;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<2, T, allocator_t> t1(dims), t2(dims);

        to_set<2,T>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<2, T> c1(t1);
            const T *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        to_set<2,T>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<2, T> c2(t2);
            const T *p = c2.req_const_dataptr();
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


template<typename T>
void to_set_test::test_3(T d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "to_set_test::test_3(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<T> allocator_t;

    try {

        index<3> i1, i2;
        i2[0] = 10; i2[1] = 2; i2[2] = 21;
        dimensions<3> dims(index_range<3>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<3, T, allocator_t> t1(dims), t2(dims);

        to_set<3,T>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<3, T> c1(t1);
            const T *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        to_set<3,T>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<3, T> c2(t2);
            const T *p = c2.req_const_dataptr();
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


template<typename T>
void to_set_test::test_4(T d) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "to_set_test::test_4(" << d << ")";
    std::string tnss = ss.str();

    typedef allocator<T> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 1; i2[1] = 4; i2[2] = 9; i2[3] = 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        dense_tensor<4, T, allocator_t> t1(dims), t2(dims);

        to_set<4,T>().perform(true, t1);
        {
            dense_tensor_rd_ctrl<4, T> c1(t1);
            const T *p = c1.req_const_dataptr();
            for(size_t i = 0; i < sz; i++) if(p[i] != 0.0) {
                std::ostringstream ss;
                ss << "Bad value at t1[" << i << "].";
                fail_test(tnss.c_str(), __FILE__, __LINE__, ss.str().c_str());
            }
            c1.ret_const_dataptr(p);
        }

        to_set<4,T>(d).perform(true, t2);
        {
            dense_tensor_rd_ctrl<4, T> c2(t2);
            const T *p = c2.req_const_dataptr();
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

    template void to_set_test::test_1<double>(double) throw(libtest::test_exception);
    template void to_set_test::test_2<double>(double) throw(libtest::test_exception);
    template void to_set_test::test_3<double>(double) throw(libtest::test_exception);
    template void to_set_test::test_4<double>(double) throw(libtest::test_exception);
    template void to_set_test::test_1<float>(float) throw(libtest::test_exception);
    template void to_set_test::test_2<float>(float) throw(libtest::test_exception);
    template void to_set_test::test_3<float>(float) throw(libtest::test_exception);
    template void to_set_test::test_4<float>(float) throw(libtest::test_exception);

} // namespace libtensor

