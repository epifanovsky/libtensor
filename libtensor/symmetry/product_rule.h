#ifndef LIBTENSOR_PRODUCT_RULE_H
#define LIBTENSOR_PRODUCT_RULE_H

#include <map>
#include <libtensor/exception.h>
#include "eval_sequence_list.h"
#include "product_table_i.h"


namespace libtensor {


template<size_t N>
class product_rule {
public:
    static const char *k_clazz;

public:
    typedef typename product_table_i::label_t label_t;
    typedef typename std::multimap<size_t, label_t>::const_iterator iterator;

    typedef typename eval_sequence_list<N>::eval_sequence_t eval_sequence_t;

private:
    eval_sequence_list<N> &m_slist; //!< Reference to list of sequences
    std::multimap<size_t, label_t> m_terms; //!< Terms in product

public:
    /** \brief Constructor
     **/
    product_rule(eval_sequence_list<N> &sl);

    /** \brief Add term to product
     **/
    void add(const eval_sequence_t &seq, label_t intr);

    /** \brief STL-style iterator over product (begin)
     **/
    iterator begin() const { return m_terms.begin(); }

    /** \brief STL-style iterator over product (begin)
     **/
    iterator end() const { return m_terms.end(); }

    /** \brief Return sequence for term pointed to by it
     **/
    const eval_sequence_t &get_sequence(iterator it) const {
        return m_slist[it->first];
    }

    /** \brief Return the number of the sequence in the sequence list
     **/
    size_t get_seqno(iterator it) const {
        return it->first;
    }

    /** \brief Return intrinsic label for term pointed to by it
     **/
    label_t get_intrinsic(iterator it) const { return it->second; }

    /** \brief Returns if product rule is empty
     **/
    bool empty() const { return m_terms.empty(); }

    /** \brief Compare with other product
     **/
    bool operator==(const product_rule<N> &pr) const;
};


/** \brief Non-equal comparison of product rules
 **/
template<size_t N>
bool operator!=(const product_rule<N> &pra, const product_rule<N> &prb) {
    return !(pra==prb);
}


template<size_t N>
const char *product_rule<N>::k_clazz = "product_rule<N>";


template<size_t N>
product_rule<N>::product_rule(eval_sequence_list<N> &sl) : m_slist(sl) {

}


template<size_t N>
void product_rule<N>::add(const eval_sequence_t &seq, label_t intr) {

    // Ignore all allowed rules in non-empty products
    if (intr == product_table_i::k_invalid && ! m_terms.empty()) return;

    // Add sequence if not available yet
    size_t seqno = m_slist.add(seq);

    // Check if sequence already exists in product
    std::multimap<size_t, label_t>::iterator it = m_terms.find(seqno);
    if (it != m_terms.end()) {
        if (it->second == intr || intr == product_table_i::k_invalid) {
            return;
        }

        if (it->second == product_table_i::k_invalid) {
            it->second = intr;
            return;
        }

    }
    m_terms.insert(std::pair<size_t, label_t>(seqno, intr));
}


template<size_t N>
bool product_rule<N>::operator==(const product_rule<N> &pr) const {

    if (pr.m_slist != m_slist) return false;

    std::map<size_t, label_t>::const_iterator it1 = m_terms.begin();
    std::map<size_t, label_t>::const_iterator it2 = pr.m_terms.begin();
    for (; it1 != m_terms.end() && it2 != m_terms.end(); it1++, it2++) {
        if (it1->first != it2->first) return false;
        if (it1->second != it2->second) return false;
    }

    if (it1 != m_terms.end() || it2 != pr.m_terms.end()) return false;

    return true;
}




} // namespace libtensor


#endif // LIBTENSOR_PRODUCT_RULE_H
