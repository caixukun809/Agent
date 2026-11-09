package com.hypersale.raglogin.repository;

import com.hypersale.raglogin.model.Tenant;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TenantRepository extends JpaRepository<Tenant, String> {}
