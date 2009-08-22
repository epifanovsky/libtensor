#ifndef PERFORMANCE_TEST_H_
#define PERFORMANCE_TEST_H_

#include <libtest.h>

namespace libtensor {

/** \brief performance tests base class 
 
 	\param Repeats number of times a performance test is repeated to get a meaningful result 
 	
 	\ingroup libtensor_performance_tests
 **/
template<size_t Repeats> 
class performance_test : public libtest::unit_test 
{
protected:
	virtual void do_calculate() = 0;
public:
	virtual void perform() throw(libtest::test_exception)
	{ 
		for ( size_t i=0; i<Repeats; i++ ) do_calculate();
	}  	
};

}

#endif /*PERFORMANCE_TEST_H_*/
