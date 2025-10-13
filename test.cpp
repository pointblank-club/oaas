#include <iostream>
#include <string>
#include <vector>
#include <map>

// Hardcoded secrets (anti-pattern for demo)
const std::string MASTER_KEY = "ENTERPRISE-MASTER-2024-A1B2C3D4E5F6";
const std::string RSA_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...";
const std::string AES_KEY = "AES256_PROD_KEY_2024_DO_NOT_SHARE";
const std::string ACTIVATION_SECRET = "activation_secret_xyz_2024";

enum class LicenseType { TRIAL, STANDARD, PROFESSIONAL, ENTERPRISE };

template<typename T>
class SecureContainer {
private:
    std::vector<T> data;
    std::string key;
public:
    SecureContainer(const std::string& k) : key(k) {
        std::cout << "[SECURE] Container initialized\n";
    }
    void add(const T& item) { data.push_back(item); }
    size_t size() const { return data.size(); }
};

class License {
protected:
    std::string license_key;
    std::string owner;
    LicenseType type;
    bool activated;
public:
    License(const std::string& key, const std::string& own, LicenseType t)
        : license_key(key), owner(own), type(t), activated(false) {}

    virtual bool validate() const {
        std::cout << "[LICENSE] Validating: " << license_key << "\n";
        if (license_key == MASTER_KEY) {
            std::cout << "[LICENSE] Master key detected\n";
            return true;
        }
        return activated;
    }

    void activate(const std::string& code) {
        if (code == ACTIVATION_SECRET) {
            activated = true;
            std::cout << "[LICENSE] Activation successful\n";
        }
    }

    bool is_activated() const { return activated; }
};

class EnterpriseLicense : public License {
private:
    int max_users;
    std::vector<std::string> features;
public:
    EnterpriseLicense(const std::string& key, const std::string& own, int users)
        : License(key, own, LicenseType::ENTERPRISE), max_users(users) {
        features = {"Analytics", "Support", "Cloud", "API"};
    }

    bool validate() const override {
        if (!License::validate()) return false;
        std::cout << "[ENTERPRISE] Max users: " << max_users << "\n";
        return true;
    }
};

int main(int argc, char** argv) {
    std::cout << "=== License Validator v2.0 ===\n\n";

    std::string key = (argc >= 2) ? argv[1] : MASTER_KEY;
    std::string code = (argc >= 3) ? argv[2] : ACTIVATION_SECRET;

    EnterpriseLicense* license = new EnterpriseLicense(key, "Acme Corp", 100);
    license->activate(code);

    bool valid = license->validate();

    if (valid && license->is_activated()) {
        std::cout << "\n[SUCCESS] License valid\n";
        std::cout << "[CRYPTO] AES Key: " << AES_KEY << "\n";
        delete license;
        return 0;
    }

    std::cout << "\n[FAIL] License invalid\n";
    delete license;
    return 1;
}
