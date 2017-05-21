#include "variant.h"

Variant::Variant()
{
    initialized = true;
}

Variant::Variant(int a)
{
    leaf = a;
}

Variant::Variant(double a)
{
    leaf = a;
}

Variant::Variant(std::__cxx11::string a)
{
    leaf = a;
}

Variant::Variant(bool a)
{
    leaf = a;
}

//void Variant::setArray()
//{
//    leaf = a;

//}

//void Variant::setObject()
//{
//    leaf = a;
//}

bool Variant::isInt()
{
    try{
        nonstd::get<int>( leaf );
        return true;
    } catch(nonstd::bad_variant_access& ){
        return false;
    }
}

bool Variant::isBool()
{
    try{
        nonstd::get<BOOL>( leaf );
        return true;
    } catch(nonstd::bad_variant_access& ){
        return false;
    }
}

bool Variant::isString()
{
    try{
        nonstd::get<std::string>( leaf );
        return true;
    } catch(nonstd::bad_variant_access& ){
        return false;
    }
}

bool Variant::isDouble()
{
    try{
        nonstd::get<double>( leaf );
        return true;
    } catch(nonstd::bad_variant_access& ){
        return false;
    }
}

bool Variant::isArray()
{
    try{
        nonstd::get<VariantList>( leaf );
        return true;
    } catch(nonstd::bad_variant_access& ){
        return false;
    }
}

bool Variant::isObject()
{
    try{
        nonstd::get<VariantMap >( leaf );
        return true;
    } catch(nonstd::bad_variant_access& ){
        return false;
    }
}

bool Variant::getBool()
{
    //try{
    BOOL mbool =  nonstd::get<BOOL >( leaf );
    return mbool.val;
    //    } catch(nonstd::bad_variant_access& ex){
    //        return false;
    //    }
}

int Variant::getInt()
{
    if(isDouble()){
        return nonstd::get<double >( leaf );
    }else{
        return nonstd::get<int >( leaf );
    }

}

std::__cxx11::string Variant::getString()
{
    std::string val =  nonstd::get<std::string >( leaf );
    return val;
}

VariantList Variant::getArray()
{
    VariantList val =  nonstd::get<VariantList >( leaf );
    return val;

}

VariantMap Variant::getMap()
{
    VariantMap val =  nonstd::get<VariantMap >( leaf );
    return val;
}

Variant &Variant::operator=(const int a)
{
    leaf = a;
    return *this;
}

Variant &Variant::operator=(const double a)
{
    leaf = a;
    return *this;
}

Variant &Variant::operator=(const char *a)
{
    leaf = std::string(a);
    return *this;
}

Variant &Variant::operator=(const bool a)
{
    BOOL b;
    b.val = a;
    leaf = b;
    return *this;
}

Variant &Variant::operator=(const VariantList a)
{
    leaf = a;
    return *this;
}

Variant &Variant::operator=(const VariantMap a)
{
    leaf = a;
    return *this;
}


bool VariantMap::hasMember(std::__cxx11::string key)
{
    return find(key)!=end();
}
