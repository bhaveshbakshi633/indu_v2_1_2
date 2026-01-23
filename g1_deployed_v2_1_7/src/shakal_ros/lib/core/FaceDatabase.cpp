#include "core/FaceDatabase.hpp"
#include "core/FaceEncoder.hpp"
#include "utils/Logger.hpp"
#include <fstream>
#include <cmath>

namespace shakal {

FaceDatabase::FaceDatabase()
    : embedding_size_(512) {
}

FaceDatabase::~FaceDatabase() = default;

bool FaceDatabase::load(const std::string& embeddings_file) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ifstream file(embeddings_file, std::ios::binary);
    if (!file.is_open()) {
        LOG_WARN("Embeddings file not found, starting with empty database");
        return true;
    }

    database_.clear();

    int num_persons = 0;
    file.read(reinterpret_cast<char*>(&num_persons), sizeof(int));
    file.read(reinterpret_cast<char*>(&embedding_size_), sizeof(int));

    for (int p = 0; p < num_persons; ++p) {
        int name_len = 0;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(int));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        int num_embeddings = 0;
        file.read(reinterpret_cast<char*>(&num_embeddings), sizeof(int));

        PersonData person;
        person.name = name;

        for (int e = 0; e < num_embeddings; ++e) {
            std::vector<float> embedding(embedding_size_);
            file.read(reinterpret_cast<char*>(embedding.data()),
                      embedding_size_ * sizeof(float));
            person.embeddings.push_back(embedding);
        }

        person.mean_embedding = computeMeanEmbedding(person.embeddings);
        database_[name] = person;
    }

    LOG_INFO("Loaded " + std::to_string(num_persons) + " persons from database");
    return true;
}

bool FaceDatabase::save(const std::string& embeddings_file) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(embeddings_file, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for writing: " + embeddings_file);
        return false;
    }

    int num_persons = static_cast<int>(database_.size());
    file.write(reinterpret_cast<const char*>(&num_persons), sizeof(int));
    file.write(reinterpret_cast<const char*>(&embedding_size_), sizeof(int));

    for (const auto& [name, person] : database_) {
        int name_len = static_cast<int>(name.length());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(int));
        file.write(name.c_str(), name_len);

        int num_embeddings = static_cast<int>(person.embeddings.size());
        file.write(reinterpret_cast<const char*>(&num_embeddings), sizeof(int));

        for (const auto& embedding : person.embeddings) {
            file.write(reinterpret_cast<const char*>(embedding.data()),
                       embedding_size_ * sizeof(float));
        }
    }

    LOG_INFO("Saved " + std::to_string(num_persons) + " persons to database");
    return true;
}

bool FaceDatabase::addPerson(const std::string& name,
                              const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        LOG_ERROR("Cannot add person with no embeddings");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Update embedding size from actual data
    embedding_size_ = static_cast<int>(embeddings[0].size());

    PersonData person;
    person.name = name;
    person.embeddings = embeddings;
    person.mean_embedding = computeMeanEmbedding(embeddings);

    database_[name] = person;
    LOG_INFO("Added person: " + name + " with " +
             std::to_string(embeddings.size()) + " embeddings (dim=" +
             std::to_string(embedding_size_) + ")");

    return true;
}

bool FaceDatabase::removePerson(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = database_.find(name);
    if (it == database_.end()) {
        LOG_WARN("Person not found: " + name);
        return false;
    }

    database_.erase(it);
    LOG_INFO("Removed person: " + name);
    return true;
}

bool FaceDatabase::updatePerson(const std::string& name,
                                 const std::vector<std::vector<float>>& embeddings) {
    return addPerson(name, embeddings);
}

MatchResult FaceDatabase::match(const std::vector<float>& embedding,
                                 float threshold) {
    MatchResult result;
    result.matched = false;
    result.similarity = 0.0f;
    result.name = "Unknown";

    if (embedding.empty()) {
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    float best_similarity = -1.0f;
    std::string best_match;

    for (const auto& [name, person] : database_) {
        float sim = FaceEncoder::cosineSimilarity(embedding, person.mean_embedding);

        if (sim > best_similarity) {
            best_similarity = sim;
            best_match = name;
        }
    }

    if (best_similarity >= threshold) {
        result.matched = true;
        result.similarity = best_similarity;
        result.name = best_match;
    } else {
        result.similarity = best_similarity;
    }

    return result;
}

std::vector<std::string> FaceDatabase::getPersonNames() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    names.reserve(database_.size());

    for (const auto& [name, _] : database_) {
        names.push_back(name);
    }

    return names;
}

size_t FaceDatabase::getPersonCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return database_.size();
}

bool FaceDatabase::hasPerson(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return database_.find(name) != database_.end();
}

std::vector<float> FaceDatabase::computeMeanEmbedding(
    const std::vector<std::vector<float>>& embeddings) {

    if (embeddings.empty()) {
        return {};
    }

    int size = static_cast<int>(embeddings[0].size());
    std::vector<float> mean(size, 0.0f);

    for (const auto& emb : embeddings) {
        for (int i = 0; i < size; ++i) {
            mean[i] += emb[i];
        }
    }

    float n = static_cast<float>(embeddings.size());
    float norm = 0.0f;

    for (int i = 0; i < size; ++i) {
        mean[i] /= n;
        norm += mean[i] * mean[i];
    }

    norm = std::sqrt(norm);
    if (norm > 1e-10) {
        for (int i = 0; i < size; ++i) {
            mean[i] /= norm;
        }
    }

    return mean;
}

}
