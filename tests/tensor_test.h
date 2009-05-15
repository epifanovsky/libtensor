#ifndef LIBTENSOR_TENSOR_TEST_H
#define LIBTENSOR_TENSOR_TEST_H

#include <libtest.h>
#include <libtensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::tensor class

	\ingroup libtensor_tests
**/
class tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Checks that requesting a non-const data pointer causes an
	//!	exception
	class test_op_chk_imm {
	private:
		bool m_ok;
	public:
		test_op_chk_imm() : m_ok(false) {}
		bool is_ok() const { return m_ok; }
		virtual void perform(tensor_i<2,int> &t) throw(exception);
	};

	//!	Sets all elements a given value
	class test_op_set_int {
	private:
		int m_val; //!< Value to set
	public:
		test_op_set_int(const int val) : m_val(val) {}
		virtual void perform(tensor_i<2,int> &t) throw(exception);
	};

	//!	Check that all elements have a given value
	class test_op_chkset_int {
	private:
		int m_val; //!< Value
		bool m_ok; //!< Indicates a positive result
	public:
		test_op_chkset_int(const int val) : m_val(val), m_ok(false) {}
		bool is_ok() const { return m_ok; }
		virtual void perform(tensor_i<2,int> &t) throw(exception);
	};

	//!	Checks that double requests for data cause an exception
	class test_op_chk_dblreq {
	private:
		bool m_ok;
	public:
		test_op_chk_dblreq() : m_ok(false) {}
		bool is_ok() const { return m_ok; }
		virtual void perform(tensor_i<2,int> &t) throw(exception);
	};

	//!	Tests the constructor
	void test_ctor() throw(libtest::test_exception);

	//!	Tests immutability
	void test_immutable() throw(libtest::test_exception);

	//!	Tests operations
	void test_operation() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_TEST_H

