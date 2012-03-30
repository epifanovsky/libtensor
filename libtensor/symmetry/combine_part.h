#ifndef LIBTENSOR_COMBINE_PART_H
#define LIBTENSOR_COMBINE_PART_H

#include "se_part.h"
#include "symmetry_element_set_adapter.h"

namespace libtensor {

/** \brief Combine multiple se_part<N, T> objects in a %symmetry element set
        to one.

    This class takes a %symmetry element set of se_part<N, T> objects and
    combines into a single se_part<N, T>. The resulting object has to produce
    the same orbits and orbit list as the full set.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class combine_part {
public:
    static const char *k_clazz; //!< Class name

    typedef se_part<N, T> se_t;

private:
    typedef symmetry_element_set_adapter<N, T, se_t> adapter_t;

private:
    adapter_t m_set; // Symmetry element set
    dimensions<N> m_pdims; //!< Partition dimensions
    block_index_space<N> m_bis; //!< Block index space

public:
    /** \brief Constructor
     **/
    combine_part(const symmetry_element_set<N, T> &set);

    /** \brief Obtain block index space
     **/
    const block_index_space<N> &get_bis() const { return m_bis; }

    /** \brief Obtain the partition dimensions of the result
     **/
    const dimensions<N> &get_pdims() const { return m_pdims; }

    /** \brief Do the
     **/
    void perform(se_t &elx);

private:
    static const block_index_space<N> &extract_bis(adapter_t &ad);
    static dimensions<N> make_pdims(adapter_t &ad);
};

} // namespace libtensor

#endif // LIBTENSOR_COMBINE_PART_H
