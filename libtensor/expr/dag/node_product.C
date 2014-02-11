#include <algorithm>
#include <libtensor/exception.h>
#include "node_product.h"

namespace libtensor {
namespace expr {


node_product::node_product(const std::string &op, size_t n,
    const std::vector<size_t> &idx) :

    node(op, n), m_idx(idx) {

    check();
}


node_product::node_product(const std::string &op, size_t n,
    const std::vector<size_t> &idx, const std::vector<size_t> &cidx) :

    node(op, n), m_idx(idx), m_cidx(cidx) {

    check();
}


void node_product::build_output_indices(std::vector<size_t> &oidx) const {

    oidx.clear();

    for(size_t i = 0; i < m_idx.size(); i++) {
        size_t idx = m_idx[i];
        if(std::find(m_cidx.begin(), m_cidx.end(), idx) != m_cidx.end()) continue;
        if(std::find(oidx.begin(), oidx.end(), idx) != oidx.end()) continue;
        oidx.push_back(idx);
    }
}


void node_product::check() {

    std::vector<size_t> oidx;
    build_output_indices(oidx);
    if(oidx.size() != get_n()) {
        throw generic_exception("libtensor::expr", "node_product", "check()",
            __FILE__, __LINE__, "Inconsistent indices.");
    }
}


} // namespace expr
} // namespace libtensor
