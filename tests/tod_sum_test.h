#ifndef LIBTENSOR_TOD_SUM_TEST_H
#define LIBTENSOR_TOD_SUM_TEST_H

#include <libtest.h>
#include <libtensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_sum class

	\ingroup libtensor_tests
**/
class tod_sum_test : public libtest::unit_test {
private:
	//!	Assigns every element its number
	class testop_set : public tod_additive<4> {
	public:
		virtual void prefetch() throw(exception) { }
		virtual void perform(tensor_i<4,double> &t) throw(exception);
		virtual void perform(tensor_i<4,double> &t, double c)
			throw(exception);
	};

	//!	Adds a constant to every element
	class testop_add : public tod_additive<4> {
	private:
		double m_v;

	public:
		testop_add(double v) : m_v(v) { }
		virtual void prefetch() throw(exception) { }
		virtual void perform(tensor_i<4,double> &t) throw(exception);
		virtual void perform(tensor_i<4,double> &t, double c)
			throw(exception);
	};

public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SUM_TEST_H

