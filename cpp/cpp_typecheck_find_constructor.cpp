/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <type_eq.h>

#include "cpp_typecheck.h"

/*******************************************************************\

Function: cpp_typecheckt::find_constructor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::find_constructor(
  const typet &start_dest_type,
  exprt &constructor_expr)
{
  constructor_expr.make_nil();

  locationt location=start_dest_type.location();
  typet dest_type(start_dest_type);
  follow_symbol(dest_type);

  if(dest_type.id()!="struct")
    return;

  const struct_typet::componentst &components=
    to_struct_type(dest_type).components();

  for(struct_typet::componentst::const_iterator
      it=components.begin();
      it!=components.end();
      it++)
  {
    const struct_typet::componentt &component=*it;
    const typet &type=component.type();

    if(type.return_type().id()=="constructor")
    {
      const irept::subt &arguments=
        type.arguments().get_sub();

      namespacet ns(context);

      if(arguments.size()==1)
      {
        const exprt &argument=(exprt &)arguments.front();
        const typet &arg_type=argument.type();

        if(arg_type.id()=="pointer" &&
           type_eq(arg_type.subtype(), dest_type, ns))
        {
          // found!
          const irep_idt &identifier=
            component.name();

          if(identifier=="")
            throw "constructor without identifier";

          constructor_expr=exprt("symbol", type);
          constructor_expr.identifier(identifier);
          constructor_expr.location()=location;
          return;
        }
      }
    }
  }
}
