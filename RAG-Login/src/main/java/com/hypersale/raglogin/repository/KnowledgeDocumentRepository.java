package com.hypersale.raglogin.repository;

import com.hypersale.raglogin.model.KnowledgeDocument;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface KnowledgeDocumentRepository extends JpaRepository<KnowledgeDocument, String> {
    List<KnowledgeDocument> findByTenantIdOrderByCreatedAtDesc(String tenantId);
    Optional<KnowledgeDocument> findByTenantIdAndId(String tenantId, String id);
}
