/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_NAMESPACE_H
#define CPROVER_NAMESPACE_H

#include "context.h"

#include <irep2.h>
#include <migrate.h>

// second: true <=> not found

class namespacet
{
public:
  const symbolt &lookup(const irep_idt &name) const
  {
    const symbolt *symbol;
    if(lookup(name, symbol))
      throw "identifier "+id2string(name)+" not found";
    return *symbol;
  }

  const symbolt &lookup(const irept &irep) const
  {
    return lookup(irep.identifier());
  }

  virtual ~namespacet()
  {
  }

  virtual bool lookup(const irep_idt &name, const symbolt *&symbol) const;
  void follow_symbol(irept &irep) const;
  void follow_macros(exprt &expr) const;

  const typet &follow(const typet &src) const;
  const type2tc follow(const type2tc &src) const
  {
    typet back = migrate_type_back(src);
    typet followed = follow(back);
    type2tc tmp;
    migrate_type(followed, tmp);
    return tmp;
  }

  namespacet(const contextt &_context)
  { context1=&_context; context2=NULL; }

  namespacet(const contextt &_context1, const contextt &_context2)
  { context1=&_context1; context2=&_context2; }

  namespacet(const contextt *_context1, const contextt *_context2)
  { context1=_context1; context2=_context2; }

  unsigned get_max(const std::string &prefix) const;

  const contextt &get_context() const
  {
    return *context1;
  }

 protected:
  const contextt *context1, *context2;
};

#endif
