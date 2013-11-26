#include <string>
#include <libtensor/core/scalar_transf_double.h>
#include "node_assign.h"
#include "node_ident.h"
#include "node_transform.h"
#include "print_node.h"

namespace libtensor {
namespace expr {


void print_node(const node &n, std::ostream &os, size_t indent) {

    std::string ind(indent, ' ');
    const node_assign *na = dynamic_cast<const node_assign*>(&n);
    const node_ident_base *ni = dynamic_cast<const node_ident_base*>(&n);
    if(ni) {
        os << ind << "(ident <" << ni->get_n() << ","
                << ni->get_t().name() << " )" << std::endl;
    } else {
        os << ind << "(" << n.get_op();
        if(n.get_op().compare("transform") == 0) {
            const node_transform_base *ntr0 =
                    dynamic_cast<const node_transform_base*>(&n);
            if(ntr0) {
                const std::vector<size_t> &p = ntr0->get_perm();
                os << "  [";
                for(size_t i = 0; i + 1 < p.size(); i++) os << p[i] << ",";
                if(p.size() > 0) os << p[p.size() - 1];
                else os << "*";
                os << "]";
                const node_transform<double> *ntrd =
                        dynamic_cast< const node_transform<double>* >(ntr0);
                if(ntrd) {
                    os << " " << ntrd->get_coeff().get_coeff();
                }
            }
        }
        os << ")" << std::endl;
    }
}


} // namespace expr
} // namespace libtensor
