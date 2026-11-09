package com.hypersale.raglogin.repository;

import com.hypersale.raglogin.model.RagMetric;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RagMetricRepository extends JpaRepository<RagMetric, Long> {
    long countByTenantId(String tenantId);
    long countByTenantIdAndEvidenceCount(String tenantId, int evidenceCount);
    List<RagMetric> findTop1000ByTenantIdOrderByCreatedAtDesc(String tenantId);
}
