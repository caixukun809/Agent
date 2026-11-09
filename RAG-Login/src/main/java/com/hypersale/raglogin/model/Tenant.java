package com.hypersale.raglogin.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "tenants")
public class Tenant {
    @Id @Column(length = 64) private String id;
    @Column(nullable = false, length = 120) private String displayName;
    @Column(nullable = false, updatable = false) private Instant createdAt;
    @PrePersist void created() { createdAt = Instant.now(); }
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getDisplayName() { return displayName; }
    public void setDisplayName(String displayName) { this.displayName = displayName; }
    public Instant getCreatedAt() { return createdAt; }
}
