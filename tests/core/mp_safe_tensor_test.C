#include <libvmm/thread.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/tensor_ctrl.h>
#include "mp_safe_tensor_test.h"

namespace libtensor {


void mp_safe_tensor_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


namespace mp_safe_tensor_test_ns {

class thread_1 : public libvmm::thread {
private:
	dense_tensor_i<1, double> &m_t1;
	dense_tensor_i<1, double> &m_t2;
	dense_tensor_i<1, double> &m_t3;
	size_t m_n;
	bool m_ok;
	std::string m_error;

public:
	thread_1(dense_tensor_i<1, double> &t1, dense_tensor_i<1, double> &t2,
		dense_tensor_i<1, double> &t3, size_t n) :
		m_t1(t1), m_t2(t2), m_t3(t3), m_n(n), m_ok(true) { }
	virtual ~thread_1() { }
	virtual void run() {
		m_ok = true;
		try {
		for(size_t i = 0; i < m_n; i++) {
			size_t sz = m_t1.get_dims().get_size();
			tensor_ctrl<1, double> c1(m_t1), c2(m_t2), c3(m_t3);
			const double *p1 = c1.req_const_dataptr();
			const double *p2 = c2.req_const_dataptr();
			double *p3 = c3.req_dataptr();

			for(size_t j = 0; j < sz; j++) p3[j] = p1[j] + p2[j];

			c3.ret_dataptr(p3);
			c2.ret_const_dataptr(p2);
			c1.ret_const_dataptr(p1);
		}
		} catch(std::exception &e) {
			m_error = e.what();
			m_ok = false;
		} catch(...) {
			m_ok = false;
		}
	}
	bool is_ok() const {
		return m_ok;
	}
	const std::string &get_error() const {
		return m_error;
	}
};

} // namespace mp_safe_tensor_test_ns


void mp_safe_tensor_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "mp_safe_tensor_test::test_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	mp_safe_tensor<2, double, allocator_t> t1(dims);
	tensor_ctrl<2, double> c1(t1);

	c1.req_prefetch();

	double *p1 = c1.req_dataptr();
	for(size_t i = 0; i < sz; i++) p1[i] = (double)i;
	c1.ret_dataptr(p1); p1 = 0;

	const double *p2 = c1.req_const_dataptr();
	for(size_t i = 0; i < sz; i++) {
		if(p2[i] != (double)i) {
			fail_test(testname, __FILE__, __LINE__,
				"Data corruption detected.");
		}
	}
	c1.ret_const_dataptr(p2);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void mp_safe_tensor_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "mp_safe_tensor_test::test_2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	mp_safe_tensor<2, double, allocator_t> t1(dims);
	tensor_ctrl<2, double> c1(t1), c2(t1);

	c1.req_prefetch();

	double *p1 = c1.req_dataptr();
	for(size_t i = 0; i < sz; i++) p1[i] = (double)i;
	c1.ret_dataptr(p1); p1 = 0;

	const double *p2 = c1.req_const_dataptr();
	const double *p3 = c2.req_const_dataptr();
	for(size_t i = 0; i < sz; i++) {
		if(p2[i] != (double)i || p3[i] != (double)i) {
			fail_test(testname, __FILE__, __LINE__,
				"Data corruption detected (1).");
		}
	}
	c1.ret_const_dataptr(p2); p2 = 0;
	c2.ret_const_dataptr(p3); p3 = 0;

	p1 = c2.req_dataptr();
	for(size_t i = 0; i < sz; i++) p1[i] = (double)(i * 2);
	c1.ret_dataptr(p1); p1 = 0;

	p2 = c1.req_const_dataptr();
	p3 = c2.req_const_dataptr();
	for(size_t i = 0; i < sz; i++) {
		if(p2[i] != (double)(i * 2) || p3[i] != (double)(i * 2)) {
			fail_test(testname, __FILE__, __LINE__,
				"Data corruption detected (2).");
		}
	}
	c1.ret_const_dataptr(p2); p2 = 0;
	c2.ret_const_dataptr(p3); p3 = 0;

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void mp_safe_tensor_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "mp_safe_tensor_test::test_3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1, i2;
	i2[0] = 999;
	dimensions<1> dims(index_range<1>(i1, i2));

	mp_safe_tensor<1, double, allocator_t> t1(dims), t2(dims), t3a(dims),
		t3b(dims), t3c(dims), t3d(dims);
	mp_safe_tensor_test_ns::thread_1 thra(t1, t2, t3a, 50);
	mp_safe_tensor_test_ns::thread_1 thrb(t1, t2, t3b, 20);
	mp_safe_tensor_test_ns::thread_1 thrc(t1, t2, t3c, 100);
	mp_safe_tensor_test_ns::thread_1 thrd(t1, t2, t3d, 60);

	thra.start();
	thrb.start();
	thrc.start();
	thrd.start();

	thra.join();
	thrb.join();
	thrc.join();
	thrd.join();

	if(!thra.is_ok()) {
		std::ostringstream ss;
		ss << "Thread A failed (" << thra.get_error() << ").";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!thrb.is_ok()) {
		std::ostringstream ss;
		ss << "Thread B failed (" << thrb.get_error() << ").";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!thrc.is_ok()) {
		std::ostringstream ss;
		ss << "Thread C failed (" << thrc.get_error() << ").";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!thrd.is_ok()) {
		std::ostringstream ss;
		ss << "Thread D failed (" << thrd.get_error() << ").";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
