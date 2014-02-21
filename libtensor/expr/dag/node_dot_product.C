#include <algorithm>
#include <libtensor/exception.h>
#include "node_dot_product.h"

namespace libtensor {
namespace expr {


const char node_dot_product::k_clazz[] = "node_dot_product";
const char node_dot_product::k_op_type[] = "dot_product";


std::vector<size_t> node_dot_product::make_idx(
    const std::vector<size_t> &idxa, const std::vector<size_t> &idxb) {

    if(idxa.size() != idxb.size()) {
        throw bad_parameter("libtensor::expr", k_clazz, "make_idx()",
            __FILE__, __LINE__, "idxa,idxb");
    }

    std::vector<size_t> idx;
    idx.insert(idx.end(), idxa.begin(), idxa.end());
    idx.insert(idx.end(), idxb.begin(), idxb.end());
    return idx;
}


std::vector<size_t> node_dot_product::make_cidx(
    const std::vector<size_t> &idxa, const std::vector<size_t> &idxb) {

    static const char method[] = "make_cidx()";

    if(idxa.size() != idxb.size()) {
        throw bad_parameter("libtensor::expr", k_clazz, method,
            __FILE__, __LINE__, "idxa,idxb");
    }

#ifdef LIBTENSOR_DEBUG
    for(size_t i = 0; i < idxa.size(); i++) {
        if(std::count(idxa.begin(), idxa.end(), idxa[i]) != 1) {
            throw bad_parameter("libtensor::expr", k_clazz, method,
                __FILE__, __LINE__, "idxa");
        }
        if(std::count(idxb.begin(), idxb.end(), idxa[i]) != 1) {
            throw bad_parameter("libtensor::expr", k_clazz, method,
                __FILE__, __LINE__, "idxb");
        }
    }
#endif // LIBTENSOR_DEBUG

    return idxa;
}


} // namespace expr
} // namespace libtensor
