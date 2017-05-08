/*******************************************************************\

Module: Type Naming for C

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <ctype.h>

#include <i2string.h>
#include <std_types.h>

#include "type2name.h"

/*******************************************************************\

Function: type2name

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string type2name(const typet &type)
{
  std::string result;
  if(type.id()=="")
    throw "Empty type encountered.";
  else if(type.id()=="empty")
    result+="V";   
  else if(type.id()=="signedbv")
    result+="S" + type.width().as_string(); 
  else if(type.id()=="unsignedbv")
    result+="U" + type.width().as_string(); 
  else if(type.is_bool()) 
    result+="B";
  else if(type.id()=="integer") 
    result+="I";
  else if(type.id()=="real") 
    result+="R";
  else if(type.id()=="complex") 
    result+="C";
  else if(type.id()=="float") 
    result+="F";
  else if(type.id()=="floatbv") 
    result+="F" + type.width().as_string();
  else if(type.id()=="fixed") 
    result+="X";
  else if(type.id()=="fixedbv") 
    result+="X" + type.width().as_string(); 
  else if(type.id()=="natural")
    result+="N";
  else if(type.id()=="pointer")
    result+="*";
  else if(type.id()=="reference")
    result+="&";
  else if(type.is_code())
  {
    const code_typet &t = to_code_type(type);
    const code_typet::argumentst arguments = t.arguments();
    result+="P(";
    for (code_typet::argumentst::const_iterator it = arguments.begin();
         it!=arguments.end();
         it++)
    {      
      result+=type2name(it->type());
      result+="'" + it->get_identifier().as_string() + "'|";
    }
    result.resize(result.size()-1);
    result+=")";
  }
  else if(type.is_array())
  {
    const array_typet &t = to_array_type(type);
    result+="ARR" + t.size().value().as_string();
  }
  else if(type.id()=="incomplete_array")
  {    
    result+="ARR?"; 
  }
  else if(type.id()=="symbol")
  {
    result+="SYM#" + type.identifier().as_string() + "#";
  }
  else if(type.id()=="struct" || 
          type.id()=="union")
  {
    if(type.id()=="struct") result +="ST";
    if(type.id()=="union") result +="UN";
    const struct_typet &t = to_struct_type(type);
    const struct_typet::componentst &components = t.components();
    result+="[";
    for(struct_typet::componentst::const_iterator it = components.begin();
        it!=components.end();
        it++)
    {            
      result+=type2name(it->type());
      result+="'" + it->name().as_string() + "'|";
    }
    result.resize(result.size()-1);
    result+="]";
  }
  else if(type.id()=="incomplete_struct")
    result +="ST?";
  else if(type.id()=="incomplete_union")
    result +="UN?";
  else if(type.id()=="c_enum")
    result +="EN" + type.width().as_string();
  else if(type.id()=="incomplete_c_enum")
    result +="EN?";
  else if(type.id()=="c_bitfield")
  {
    result+="BF" + type.size().as_string(); 
  }
  else
  {
    throw (std::string("Unknown type '") + 
           type.id().as_string() + 
           "' encountered."); 
  }
    
  if(type.has_subtype())
  {
    result+="{";
    result+=type2name(type.subtype());    
    result+="}";
  }

  if(type.has_subtypes())
  {
    result+="$";
    forall_subtypes(it, type)
    {      
      result+=type2name(*it);
      result+="|";      
    }
    result.resize(result.size()-1);
    result+="$";
  }
  
  return result;
}

