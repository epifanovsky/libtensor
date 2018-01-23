#ifndef LIBTEST_UNIT_TEST_FACTORY_H
#define LIBTEST_UNIT_TEST_FACTORY_H

#include "unit_test.h"

namespace libtest {


/** \brief Interface for unit test factories
    \ingroup libtest
 **/
class unit_test_factory_i {
public:
    /** \brief Creates an instance of a test
     **/
    virtual unit_test *create_instance() = 0;

};


/** \brief Instantiates unit_test classes via new
    \param T unit_test class
    \ingroup libtest
 **/
template<class T>
class unit_test_factory : public unit_test_factory_i {
public:
    /** \brief Creates an instance of the test

        The method uses new to create an instance of the test class.
        The user of this method has to delete the instance after.    
     **/
    virtual unit_test *create_instance() {

        return new T;
    }

};


} // namespace libtest

#endif // LIBTEST_UNIT_TEST_FACTORY_H

