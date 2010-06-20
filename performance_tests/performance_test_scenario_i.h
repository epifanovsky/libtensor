#ifndef PERFORMANCE_TEST_SCENARIO_I_H_
#define PERFORMANCE_TEST_SCENARIO_I_H_

#include <vector>
#include <string>
#include <utility>
#include <libtest/libtest.h>
#include <libtensor/exception.h>
#include <libtensor/core/out_of_bounds.h>

namespace libtensor {
/** \brief Performance test scenario interface

 	A performance scenario is supposed to store multiple tests which should be
 	executed one after the other.

 	\ingroup libtensor_performance_tests

 **/
class performance_test_scenario_i {
	typedef struct test_data {
		std::string m_name; //!< name of test
		std::string m_desc; //!< description of test
		libtest::unit_test_factory_i* m_utf;

		test_data( const char* name, const char* desc,
				libtest::unit_test_factory_i* utf )
			: m_name(name), m_desc(desc), m_utf(utf) {}
	} test_t;
	std::vector<test_t> m_sc_tests;
protected:
	void add_test( const char* name, const char* desc,
			libtest::unit_test_factory_i& utf );
public:
	//! number of tests in the scenario
	size_t size() {	return m_sc_tests.size(); }
	//! name of i-th test
	std::string& test_name( size_t );
	//! description of i-th test
	std::string& test_description( size_t );
	//! unit_test_factory for i-th test
	libtest::unit_test_factory_i& test( size_t );

};

inline std::string&
performance_test_scenario_i::test_name( size_t i )
{
	if ( i >= m_sc_tests.size() )
		throw out_of_bounds("libtensor","performance_test_scenario_i",
			"test_name( size_t )",__FILE__,__LINE__,"Invalid index.");

	return m_sc_tests[i].m_name;

}
inline std::string&
performance_test_scenario_i::test_description( size_t i )
{
	if ( i >= m_sc_tests.size() )
		throw out_of_bounds("libtensor","performance_test_scenario_i",
			"test_name( size_t )",__FILE__,__LINE__,"Invalid index.");

	return m_sc_tests[i].m_desc;

}
inline libtest::unit_test_factory_i&
performance_test_scenario_i::test( size_t i)
{
	if ( i >= m_sc_tests.size() )
		throw out_of_bounds("libtensor","performance_test_scenario_i",
			"test( size_t )",__FILE__,__LINE__,"Invalid index.");

	return *(m_sc_tests[i].m_utf);
}

inline void
performance_test_scenario_i::add_test( const char* name, const char* desc,
		libtest::unit_test_factory_i& utf )
{
	m_sc_tests.push_back(test_t(name,desc,&utf));
}

}
#endif /*PERFORMANCE_TEST_SCENARIO_I_H_*/
