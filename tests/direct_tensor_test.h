#ifndef LIBTENSOR_DIRECT_TENSOR_TEST_H
#define LIBTENSOR_DIRECT_TENSOR_TEST_H

#include <libtest.h>
#include "direct_tensor.h"

namespace libtensor {

/**	\brief Tests the libtensor::direct_tensor class

	\ingroup libtensor_tests
**/
class direct_tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Direct tensor operation stub
	class test_op : public direct_tensor_operation<int> {
	private:
		bool m_ok;
	public:
		test_op() : m_ok(false) {}
		bool is_ok() const { return m_ok; }
		virtual void prefetch() throw(exception) {}
		virtual void perform(tensor_i<int> &t) throw(exception);
	};

	//!	Sets every tensor element to its sequential number
	class test_op_set : public direct_tensor_operation<int> {
	private:
		int m_count;
	public:
		test_op_set() : m_count(0) {}
		int get_count() const { return m_count; }
		virtual void prefetch() throw(exception) {}
		virtual void perform(tensor_i<int> &t) throw(exception);
	};

	//!	Checks if tensor elements are set correctly
	class test_op_chk_set : public direct_tensor_operation<int> {
	private:
		bool m_ok;
	public:
		test_op_chk_set() : m_ok(false) {}
		bool is_ok() const { return m_ok; }
		virtual void prefetch() throw(exception) {}
		virtual void perform(tensor_i<int> &t) throw(exception);
	};

	//!	Tests the constructor
	void test_ctor() throw(libtest::test_exception);

	//!	Tests buffering
	void test_buffering() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_TENSOR_TEST_H

