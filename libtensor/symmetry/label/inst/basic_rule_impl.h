#ifndef LIBTENSOR_BASIC_RULE_IMPL_H
#define LIBTENSOR_BASIC_RULE_IMPL_H

namespace libtensor {

template<size_t N>
bool operator==(const basic_rule<N> &br1, const basic_rule<N> &br2) {

    typedef typename basic_rule<N>::label_set_t label_set_t;

    for (register size_t i = 0; i < N; i++) {
        if (br1[i] != br2[i]) return false;
    }

    const label_set_t &ls1 = br1.get_target(), &ls2 = br2.get_target();

    if (ls1.size() != ls2.size()) return false;
    for (typename label_set_t::const_iterator it1 = ls1.begin(),
            it2 = ls2.begin(); it1 != ls1.end(); it1++, it2++) {
        if (*it1 != *it2) return false;
    }

    return true;
}

} // namespace libtensor

#endif // LIBTENSOR_BASIC_RULE_IMPL_H
