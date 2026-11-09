package com.hypersale.raglogin.service;

import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.DocumentVisibility;
import com.hypersale.raglogin.model.KnowledgeDocument;
import com.hypersale.raglogin.repository.KnowledgeDocumentRepository;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.List;

@Service
public class DocumentAccessService {
    private final KnowledgeDocumentRepository documents;

    public DocumentAccessService(KnowledgeDocumentRepository documents) { this.documents = documents; }

    public List<KnowledgeDocument> accessible(AppUser user) {
        return documents.findByTenantIdOrderByCreatedAtDesc(user.getTenantId()).stream()
                .filter(document -> canRead(user, document)).toList();
    }

    public boolean canRead(AppUser user, KnowledgeDocument document) {
        if (!user.getTenantId().equals(document.getTenantId())) return false;
        if (user.getRole().name().equals("ADMIN")) return true;
        if (document.getOwnerUserId().equals(user.getId())) return true;
        if (document.getVisibility() == DocumentVisibility.TENANT_PUBLIC) return true;
        return document.getVisibility() == DocumentVisibility.SELECTED_USERS
                && document.getGrantedUserIds().contains(user.getId());
    }

    public Path tenantRoot(String tenantId, java.nio.file.Path configuredRoot) {
        Path root = configuredRoot.toAbsolutePath().normalize();
        Path tenantRoot = root.resolve(tenantId).normalize();
        if (!tenantRoot.startsWith(root)) throw new IllegalArgumentException("租户目录无效");
        return tenantRoot;
    }
}
