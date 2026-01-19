#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace shakal {

struct PersonData {
    std::string name;
    std::vector<std::vector<float>> embeddings;
    std::vector<float> mean_embedding;
};

struct MatchResult {
    std::string name;
    float similarity;
    bool matched;
};

class FaceDatabase {
public:
    FaceDatabase();
    ~FaceDatabase();

    bool load(const std::string& embeddings_file);
    bool save(const std::string& embeddings_file);

    bool addPerson(const std::string& name,
                   const std::vector<std::vector<float>>& embeddings);
    bool removePerson(const std::string& name);
    bool updatePerson(const std::string& name,
                      const std::vector<std::vector<float>>& embeddings);

    MatchResult match(const std::vector<float>& embedding,
                      float threshold = 0.5f);

    std::vector<std::string> getPersonNames() const;
    size_t getPersonCount() const;
    bool hasPerson(const std::string& name) const;

private:
    std::unordered_map<std::string, PersonData> database_;
    mutable std::mutex mutex_;
    int embedding_size_;

    std::vector<float> computeMeanEmbedding(
        const std::vector<std::vector<float>>& embeddings);
};

}
