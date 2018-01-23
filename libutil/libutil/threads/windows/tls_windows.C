#include "tls_windows.h"

namespace libutil {


mutex tls_windows_destructor_list::m_lock;
std::vector<tls_windows_destructor_list::destructor_fn>
    tls_windows_destructor_list::m_lst;


} // namespace libutil
