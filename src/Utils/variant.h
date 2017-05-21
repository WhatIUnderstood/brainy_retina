#ifndef VARIANT_H
#define VARIANT_H

#include <vector>
#include <map>
#include "nonstd/variant.hpp"

class Variant;

typedef std::vector<Variant> VariantList;
//typedef std::map<std::string,Variant> VariantMap;

class VariantMap: public std::map<std::string,Variant>{
public:
    bool hasMember(std::string key);
};

class Variant
{
public:
    Variant();
    Variant(int a);
    Variant(double a);
    Variant(std::string a);
    Variant(bool a);
    //    void setArray();
    //    void setObject();
    bool isInt();
    bool isBool();
    bool isString();
    bool isDouble();
    bool isArray();
    bool isObject();

    bool getBool();
    int getInt();
    std::string getString();
    VariantList getArray();
    VariantMap getMap();

    Variant& operator= (const int a);
    Variant& operator= (const double a);
    Variant& operator= (const char* a);
    Variant& operator= (const bool a);
    Variant& operator= (const VariantList a);
    Variant& operator= (const VariantMap a);


    //Variant();
    //nonstd::variant< char, int, long, std::string > var;
private:

    //    enum INTERNAL_TYPE{
    //        OBJECT,
    //        ARRAY,
    //        LEAF,
    //        NONE
    //    };

    struct BOOL{ //Used to avoid conflict between bool and int
        bool val;
    };

    bool initialized;

    //    std::vector<Variant> variants_list;
    //    std::map<std::string,Variant> variants_map;
    nonstd::variant<int,std::string,double,BOOL,VariantList,VariantMap> leaf;
    //    INTERNAL_TYPE type;

    //    template <typename T> void assign(T value){
    //        type = LEAF;
    //        leaf = value;
    //    }
};

#endif // VARIANT_H
