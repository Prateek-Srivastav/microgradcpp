#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <format>
#include <sstream>
#include <ctime>

using namespace std;


class Value {;
public:
    double data;
    double grad;
    string _op;
    vector<Value> _children;
    function<void()> _backward;
    
    // default constructor
    Value() : data(0.0), grad(0.0), _op(""), _children(), _backward([](){}) {}

    // constructor
    Value(double data, const string& op = "") : data(data), grad(0.0), _op(op), _children(), _backward([](){}) {}

    // overload + operator
    Value operator+(Value& other) {
        Value out(this->data + other.data, "+");
        out._children.push_back(*this);
        out._children.push_back(other);
        out._backward = [this, &other, &out]() mutable
        {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        
        return out;
    }

    // friend function to overload the + operator for dtype + Value
    template <typename T>
    friend Value operator+(const T& lhs, const Value& rhs) {
        Value temp(lhs);
        return temp + rhs;
    }

     // overload + operator for Value + other dtypes
    template <typename T>
    Value operator+(const T& other) {
        Value temp(other); // convert dtype to Value
        return *this + temp; // reuse existing operator+
    }
    
    // overload - operator
    Value operator-(Value& other) {
        Value out = *this + (other * -1);
        return out;
    }

    // overload - operator
    template <typename T>
    Value operator-(const T& other) {
        Value temp(other);
        Value out = *this + (temp * -1);
        return out;
    }


    // overload * operator
    Value operator*(Value& other) {
        Value out(this->data * other.data, "*");
        out._children.push_back(*this);
        out._children.push_back(other);
        out._backward = [this, &other, &out]() mutable
        {  
            this->grad += other.data * out.grad; 
            other.grad += this->data * out.grad;
        };
        return out;
    }

    template <typename T>
    Value operator*(const T& other) {
        Value temp = other; // convert dtype to Value
        return *this * temp;
    }

    // friend function to overload the * operator for dtype * Value
    template <typename T>
    friend Value operator*(const T& lhs, const Value& rhs) {
        Value temp = lhs;
        return temp * rhs;
    }

    // overload the pow function
    Value pow(const double& other) {
        stringstream op;
        op << "**" << other;

        Value out(std::pow(this->data, other), op.str());
        out._children.push_back(*this);
    
        out._backward = [this, &other, &out]() mutable {
            // cout << "out" << out << endl;
            this->grad += (other * std::pow(this->data, other - 1)) * out.grad;
            // this->grad += 2;
            cout << "this" << *this << endl;
        };
        return out;
    }

    // overload / operator
    Value operator/(Value& other) {
        Value out(this->data / other.data, "/");
        out._children.push_back(*this);
        out._children.push_back(other);
        out._backward = [this, &other, &out]() mutable {
            this->grad = (1 / other.data) * out.grad;
            other.grad = (-this->data / std::pow(other.data, 2)) * out.grad;
        };
        return out;
    }

    // template <typename T>
    // Value operator/(const T& other) {
    //     Value temp(other); // Ccnvert dtype to Value
    //     return *this / temp;
    // }

    // // friend function to overload the / operator for dtype / Value
    // template <typename T>
    // friend Value operator/(const T& lhs, Value& rhs) {
    //     *lhs = Value(*lhs);
    //     return lhs / rhs;
    // }

    Value tanh()  {
        double x = this->data;
        double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
        Value out(t, "tanh");
        out._children.push_back(*this);
        out._backward = [this, t, &out]() mutable {
            this->grad += (1-t*t) * out.grad;     // diffrentiation of tanh is 1 - tanh**2
        };
        return out;
    }

    Value exp() {
        double ex  = std::exp(this->data);
        Value out(ex, "exp");
        out._children.push_back(*this);
        cout << "outs" << out << endl;
        out._backward = [this, ex, &out]() mutable
        {
            this->grad += out.data * out.grad;
            // this->grad += 3;
            // cout << "exp" << *this << endl;
        };
        return out;
    }

    void backward() {
        vector<Value*> topo;
        unordered_set<Value*> visited;
        function<void(Value*)> build_topo = [&](Value* v) {
            if (visited.count(v) == 0) {
                visited.insert(v);
                for (auto& child : v->_children)
                    build_topo(&child);
                topo.push_back(v);
            }
        };

        build_topo(this);

        this->grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
            // cout << **it << endl;
        }
    }

    // overload << operator for printing Value
    friend ostream& operator<<(ostream& os, const Value& value) {
         os << "Value(data=" << value.data << ", grad=" << value.grad << ", _op='" << value._op << "', _children=[";
        for (size_t i = 0; i < value._children.size(); ++i) {
            os << value._children[i].data;
            if (i < value._children.size() - 1) {
                os << ", ";
            }
        }
        os << "])";
        return os;
    }
};

class Neuron {
public:
    vector<Value> w;
    Value b;
    Neuron(int nin) {
        srand(static_cast<unsigned int>(rand() * time(nullptr))); // setting seed according to current time to get random number every time
        w.reserve(nin);
        for (int i = 0; i < nin; i++)
            w.emplace_back((rand() / double(RAND_MAX)) * 2.0 - 1.0);

        b = (rand() / double(RAND_MAX)) * 2.0 - 1.0;
    }

    Value operator()(const vector<double>& x) const {
        Value act = b.data;
        for (int i = 0; i < w.size(); i++) 
            act.data += w[i].data * x[i];
        return act.tanh();
        
    }
};

class Layer {
public:
    vector<Neuron> neurons;
    Layer(int nin, int nout) {
        for (int i = 0; i < nout; i++) 
            neurons.emplace_back(nin);
    }

    vector<Value> operator()(const vector<double>& x) const {
        vector<Value> outs;
        outs.reserve(neurons.size());
        for (const Neuron& n : neurons) 
            outs.push_back(n(x));
        return outs;
    }
};

class MLP {
public:
    vector<Layer> layers;
    MLP(int nin, const vector<int>& nouts) {
        vector<int> sz;
        sz.push_back(nin);
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (int i = 0; i < nouts.size(); i++)
            layers.emplace_back(Layer(sz[i], sz[i + 1]));
    }

    vector<Value> operator()(const vector<double>& x) {
        vector<double> xValue = x;
        vector<Value> out;

        for (const Layer& layer : layers)
            out = layer(xValue);
        return out;
    }
};
int main() {
    vector<double> x{2.0, 3.0, -1.0};
    vector<int> nouts{4, 4, 1};
    MLP z = MLP(3, nouts);

    vector<vector<double>> xs{
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0},
    };

    vector<double> ys{1.0, -1.0, -1.0, 1.0}; // desired targets

    vector<Value> ypred;
    for (int i = 0; i < xs.size(); i++){
        Value pred = z(xs[i])[0];
        ypred.push_back(pred);
    }

    // for (auto pred : ypred)
    //     cout << pred << endl;

    Value loss = 0.0;
    for (int i = 0; i < ys.size(); i ++){
        Value l = Value(ys[i]) - ypred[i];
        Value final_l = l.pow(2);
        loss = loss + (final_l);
    }
    cout << "loss = " << loss << endl;

    // loss.backward();

    // cout << "weight" << z.layers[0].neurons[0].w[0] << endl;

    // cout << "z = " << z(xs[i])[0] << endl;
    // Create Value instances
    // Value a(2.0);
    // Value b(-3.0);
    // Value c(10.0);
    // Value d(0);
    // Value e(0);
    // Value f(-2.0);
    // Value L(0);
    // e = a * b;
    // d = e + c;
    // L = d * f;

    // // assigning grads manually
    // // dL/df = d, dL/dd = f
    // f.grad = 4.0;
    // d.grad = -2.0;

    
    // // dL/dc = (dd/dc=1.0) * dL/dd 'chain rule'
    // c.grad = -2.0;
    // e.grad = -2.0;

    // // dL/da = (dL/de=-2.0) * (de/da=b) 'chain rule'
    // a.grad = -2.0 * -3.0;
    // b.grad = -2.0 * 2.0;

    // cout << "a=" << a << endl;
    // cout << "b=" << b << endl;
    // cout << "e=" << e << endl;
    // cout << "c=" << c << endl;
    // cout << "d=" << d << endl;
    // cout << "f=" << f << endl;
    // cout << "L=" << L << endl;

    // inputs x1, x2
    Value x1 = 2.0;
    Value x2 = 0.0;
    // weights w1, w2
    Value w1 = -3.0;
    Value w2 = 1.0;
    // bias of the neuron
    Value b = 6.8813735870195432;
    // x1*w1 + x2*w2 + b
    Value x1w1 = x1 * w1;
    Value x2w2 = x2 * w2;
    Value x1w1x2w2 = x1w1 + x2w2;
    Value n = x1w1x2w2 + b;
    Value o = n.tanh();
    // instead of direct tanh, let's breakdown
    // Value a1 = 1;
    // Value a2 = 2;
    // Value n1 = a2.pow(-1);
    // Value neg1 = -1;
    // Value n3 = 3;
    // Value e = (a2*n).exp();
    // Value num = e - a1;
    // Value den = e + a1;
    // Value den2 = den.pow(-1);
    // Value n4 = n3.pow(-1);
    // Value o = num / den;
    // Value num = n2 * 3;
    // Value den = e + 1;

    // Value denom = pow(den, -1);
    // Value o = num / den;

    // o.grad = 1.0;     // o.grad is initialized as 0 and do/do is 1
    // o._backward();
    // n2._backward();
    // n1._backward();
    // b._backward();
    // x1w1x2w2._backward();
    // x2w2._backward();
    // x1w1._backward();
    // Value x = (x1 + x1);
    // x.backward();
    o.backward();


    cout << "x1 = " << x1 << endl;
    cout << "w1 = " << w1 << endl;
    cout << "x2 = " << x2 << endl;
    cout << "w2 = " << w2 << endl;
    cout << "x1w1 = " << x1w1 << endl;
    cout << "x2w2 = " << x2w2 << endl;
    cout << "x1w1x2w2 = " << x1w1x2w2 << endl;
    cout << "b = " << b << endl;
    cout << "n = " << n << endl;
    cout << "o = " << o << endl;

    return 0;
}
// auto init = []()
// {
//     ios::sync_with_(0);
//     cin.tie(0);
//     cout.tie(0);
//     return 'c';
// }();
