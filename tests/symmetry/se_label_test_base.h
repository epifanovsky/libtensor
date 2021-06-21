#ifndef LIBTENSOR_SE_LABEL_TEST_BASE_H
#define LIBTENSOR_SE_LABEL_TEST_BASE_H

#include <sstream>
#include <vector>
#include <libtest/unit_test.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/symmetry/se_label.h>


namespace libtensor {

/** \brief Base class for libtensor::se_label related tests

    \ingroup libtensor_tests_sym
 **/
class se_label_test_base : public libtest::unit_test {
protected:

    /** \brief Creates a point group table with given ID
        \param id to access the table

        Valid values of table ID are C2v (abelian point group) and S6.
     **/
    void setup_pg_table(
            const std::string &id);

    /** \brief Removes point group table with given ID
        \param id to access the table
     **/
    void clear_pg_table(
            const std::string &id);

    /** \brief Checks the allowed blocks of a symmetry element against a
            reference
        \param testname Name of the test for which the check is performed.
        \param sename Identifier of the symmetry element.
        \param se Symmetry element.
        \param expected The reference data

        The strings testname and sename are used to generate more meaningful
        error messages.
        The reference \c expected needs to have one element for each block
        in the symmetry element. If the reference is true the respective block
        is expected to be allowed.
     **/
    template<size_t N>
    void check_allowed(const char *testname, const char *sename, 
            const se_label<N, double> &se, const std::vector<bool> &expected)
       ;
};

template<size_t N>
void se_label_test_base::check_allowed(
        const char *testname, const char *sename,
        const se_label<N, double> &se, const std::vector<bool> &expected)
    {

    const block_labeling<N> &bl = se.get_labeling();
    const dimensions<N> &bidims = bl.get_block_index_dims();

    if (bidims.get_size() != expected.size())
        throw;

    abs_index<N> ai(bidims);
    do {

        if (se.is_allowed(ai.get_index()) != expected[ai.get_abs_index()]) {
            std::ostringstream oss;
            oss << (expected[ai.get_abs_index()] ? "!" : "")
                << sename << ".is_allowed(" << ai.get_index() << ")";
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }

    } while (ai.inc());

}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_TEST_BASE_H

