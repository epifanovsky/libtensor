#ifndef TOD_ADD_P_TEST_H
#define TOD_ADD_P_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief First performance test of the libtensor::tod_add class
  
 	Test performance of
 	\f[ A_{pqrs} = A_{pqrs} + 2.0 B_{pqrs} \f]

	\ingroup libtensor_tests
**/
class tod_add_p1_test 
	: public libtest::unit_test, public timings<tod_add_p1_test> 
{
	friend timings<tod_add_p1_test>;
	static const char* k_clazz;
		
	void do_calculate() throw(libtest::test_exception);
public:
	virtual void perform() throw(libtest::test_exception);
};

/**	\brief Second performance test of the libtensor::tod_add class

 	Test performance of
 	\f[ A_{pqrs} = A_{pqrs} + 2.0 B_{srqp} \f]

	\ingroup libtensor_tests
**/
class tod_add_p2_test 
	: public libtest::unit_test, public timings<tod_add_p1_test> 
{
	friend timings<tod_add_p2_test>;
	
	static const char* k_clazz;
		
	void do_calculate() throw(libtest::test_exception);
public:
	virtual void perform() throw(libtest::test_exception);
};

/**	\brief Third performance test of the libtensor::tod_add class

 	Test performance of
 	\f[ A_{pqrs} = A_{pqrs} + 2.0 B_{sqrp} \f]

	\ingroup libtensor_tests
**/
class tod_add_p3_test 
	: public libtest::unit_test, public timings<tod_add_p1_test> 
{
	friend timings<tod_add_p3_test>;

	static const char* k_clazz;
		
	void do_calculate() throw(libtest::test_exception);
public:
	virtual void perform() throw(libtest::test_exception);
};

}

#endif // TOD_ADD_P1_TEST_H

