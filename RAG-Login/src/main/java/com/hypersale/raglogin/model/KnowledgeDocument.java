package com.hypersale.raglogin.model;

import jakarta.persistence.CollectionTable;
import jakarta.persistence.Column;
import jakarta.persistence.ElementCollection;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;

import java.time.Instant;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "knowledge_documents")
public class KnowledgeDocument {
    @Id
    @Column(length = 36)
    private String id;

    @Column(nullable = false, length = 80)
    private String tenantId;

    @Column(nullable = false)
    private Long ownerUserId;

    @Column(nullable = false, length = 255)
    private String originalName;

    @Column(nullable = false, length = 2048)
    private String storedPath;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 24)
    private DocumentVisibility visibility;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "document_user_grants", joinColumns = @JoinColumn(name = "document_id"))
    @Column(name = "user_id", nullable = false)
    private Set<Long> grantedUserIds = new HashSet<>();

    @Column(nullable = false, length = 24)
    private String indexStatus = "PENDING";

    @Column(nullable = false, updatable = false)
    private Instant createdAt;

    @PrePersist
    void created() { createdAt = Instant.now(); }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getTenantId() { return tenantId; }
    public void setTenantId(String tenantId) { this.tenantId = tenantId; }
    public Long getOwnerUserId() { return ownerUserId; }
    public void setOwnerUserId(Long ownerUserId) { this.ownerUserId = ownerUserId; }
    public String getOriginalName() { return originalName; }
    public void setOriginalName(String originalName) { this.originalName = originalName; }
    public String getStoredPath() { return storedPath; }
    public void setStoredPath(String storedPath) { this.storedPath = storedPath; }
    public DocumentVisibility getVisibility() { return visibility; }
    public void setVisibility(DocumentVisibility visibility) { this.visibility = visibility; }
    public Set<Long> getGrantedUserIds() { return grantedUserIds; }
    public void setGrantedUserIds(Set<Long> grantedUserIds) { this.grantedUserIds = grantedUserIds; }
    public String getIndexStatus() { return indexStatus; }
    public void setIndexStatus(String indexStatus) { this.indexStatus = indexStatus; }
    public Instant getCreatedAt() { return createdAt; }
}
