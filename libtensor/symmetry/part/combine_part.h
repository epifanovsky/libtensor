#ifndef LIBTENSOR_COMBINE_PART_H
#define LIBTENSOR_COMBINE_PART_H

#include "../se_part.h"
#include "../symmetry_element_set_adapter.h"

namespace libtensor {

/** \brief Combine multiple se_part<N, T> objects in a %symmetry element set
        to one

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
    static const block_index_space<N> extract_bis(adapter_t &ad);
    static dimensions<N> make_pdims(adapter_t &ad);
};


template<size_t N, typename T>
const char *combine_part<N, T>::k_clazz = "combine_part<N, T>";

template<size_t N, typename T>
combine_part<N, T>::combine_part(const symmetry_element_set<N, T> &set) :
    m_set(set), m_bis(extract_bis(m_set)), m_pdims(make_pdims(m_set)) {
}

template<size_t N, typename T>
void combine_part<N, T>::perform(se_t &elx) {

    if (el.get_pdims() != m_pdims) {
        throw bad_parameter(g_ns, k_clazz, "perform(se_t &)",
                __FILE__, __LINE__, "pdims");
    }

    for (adapter_t::iterator it = m_set.begin(); it != m_set.end(); it++) {

        const se_t &e1 = m_set.get_elem(it);
        const dimensions<N> pdims = e1.get_pdims();

        abs_index<N> aix(m_pdims);
        do {
            const index<N> &ix = aix.get_index();

            index<N> i1;
            for (register size_t i = 0; i < N; i++) {
                if (pdims[i] != 1) { i1[i] = ix[i]; }
            }

            if (e1.is_forbidden(i1)) {
                elx.mark_forbidden(ix);
                continue;
            }

            index<N> i2 = e1.get_direct_map(i1);
            if (i1 == i2) continue;

            bool sign = e1.get_sign(i1, i2);

            for (size_t i = 0; i < N; i++) {
                if (pdims[i] == 1) { i2[i] = ix[i]; }
            }

            if (elx.is_forbidden(ix)) {
                elx.mark_forbidden(i2);
            }
            else {
                elx.add_map(ix, i2, sign);
            }
        } while (ai1.inc());
    }

}

template<size_t N, typename T>
static const block_index_space<N> &combine_part<N, T>::extract_bis(
        adapter_t &ad) {

    adapter_t::iterator it = ad.begin();
    const block_index_space<N> &bis = ad.get_elem(it).get_bis();
    it++;
    for (; it != ad.end(); it++) {

        if (! bis.equals(ad.get_elem(it).get_bis())) {
            throw bad_symmetry(g_ns, k_clazz, "extract_bis(adapter_t &)",
                    __FILE__, __LINE__, "bis");
        }
    }

    return bis;
}

template<size_t N, typename T>
static dimensions<N> combine_part<N, T>::make_pdims(adapter_t &ad) {

    index<N> i1, i2;
    for (adapter_t::iterator it = ad.begin(); it != ad.end(); it++) {

        const se_t &el = ad.get_elem(it);
        const dimensions<N> &pdims = el.get_pdims();

        for (size_t i = 0; i < N; i++) {
            if (pdims[i] != 1) {
                if ((i2[i] != 0) && (pdims[i] - 1 != i2[i])) {
                    throw bad_symmetry(g_ns, k_clazz,
                            "make_pdims(adapter_t &)", __FILE__,
                            __LINE__, "pdims");
                }
                i2[i] = pdims[i] - 1;
            }
        }
    }

    return dimensions<N>(index_range<N>(i1, i2));
}

} // namespace libtensor

#endif // LIBTENSOR_COMBINE_PART_H
